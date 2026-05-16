"""Capture loop entrypoint with hotplug-aware state machine.

States:
    SEARCHING   no live camera; emits a synthetic black frame at ~target FPS.
    CAPTURING   has a live cv2.VideoCapture handed to a CaptureReader thread;
                forwards each new frame (letterboxed) into SHM.

Run with:
    uv run capture-run                  # uses ./config.yaml
    uv run capture-run --config path
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from .capture_session import CaptureReader, open_camera
from .device_manager import enumerate_devices, select_device
from .frame_normalizer import letterbox
from .hotplug_watcher import HotplugEvent, HotplugWatcher
from .shm_writer import FrameSHM

log = logging.getLogger("capture")

READ_TIMEOUT_SEC = 0.5  # HANDOFF 4.4: 500ms
RESCAN_INTERVAL_SEC = 1.0


class State(Enum):
    SEARCHING = "SEARCHING"
    CAPTURING = "CAPTURING"


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _build_black_frame(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _try_open(cam_cfg: dict, fmt_cfg: dict):
    devs = enumerate_devices()
    chosen = select_device(
        devs,
        cam_cfg.get("preferred", []),
        cam_cfg.get("fallback", "any"),
        format_hint=fmt_cfg,
    )
    if chosen is None:
        return None
    return open_camera(
        chosen,
        fourcc_priority=fmt_cfg["fourcc_priority"],
        width=fmt_cfg["width"],
        height=fmt_cfg["height"],
        fps=fmt_cfg["fps"],
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--print-fps-every",
        type=float,
        default=2.0,
        help="seconds between FPS log lines (0 disables)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cam_cfg = cfg["camera"]
    fmt_cfg = cam_cfg["format"]
    out_cfg = cam_cfg["output"]
    shm_cfg = cfg["shm"]
    target_w, target_h = out_cfg["target"]
    target_fps = float(fmt_cfg["fps"])
    frame_period = 1.0 / target_fps if target_fps > 0 else 1.0 / 30

    # initial inventory log
    initial = enumerate_devices()
    log.info("startup device inventory (%d):", len(initial))
    for d in initial:
        log.info("  %-58s %s vid:pid=%s:%s", d.dev_path, d.by_id or "(no by-id)", d.vid, d.pid)

    shm = FrameSHM.create(
        name=shm_cfg["name"],
        frame_w=target_w,
        frame_h=target_h,
        channels=3,
        pixel_format=shm_cfg.get("pixel_format", "BGR"),
    )
    log.info("SHM ready: name=%s shape=(%d,%d,3)", shm_cfg["name"], target_h, target_w)

    watcher = HotplugWatcher()
    watcher.start()

    black = _build_black_frame(target_w, target_h)

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        log.info("signal %d received, shutting down", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    state = State.SEARCHING
    reader: CaptureReader | None = None
    last_frame_ts: float = -1.0
    last_scan_ts: float = -RESCAN_INTERVAL_SEC  # force first scan immediately
    fps_window = 0
    fps_t0 = time.monotonic()

    def _enter_searching(reason: str) -> None:
        nonlocal state, reader, last_frame_ts
        log.info("-> SEARCHING (%s)", reason)
        if reader is not None:
            reader.stop()
            reader = None
        last_frame_ts = -1.0
        state = State.SEARCHING

    def _enter_capturing(new_reader: CaptureReader) -> None:
        nonlocal state, reader
        log.info(
            "-> CAPTURING dev=%s by_id=%s vid:pid=%s:%s",
            new_reader.device.dev_path,
            new_reader.device.by_id,
            new_reader.device.vid,
            new_reader.device.pid,
        )
        new_reader.start()
        reader = new_reader
        state = State.CAPTURING

    try:
        while not stop["flag"]:
            now = time.monotonic()

            # 1) drain hotplug events. they may force SEARCHING or trigger an
            #    immediate rescan in SEARCHING.
            events: list[HotplugEvent] = watcher.drain(timeout=0.0)
            had_add = False
            for ev in events:
                if ev.action == "remove":
                    if state is State.CAPTURING and reader is not None and ev.dev_path == reader.device.dev_path:
                        _enter_searching(f"active device {ev.dev_path} removed")
                elif ev.action == "add":
                    had_add = True

            # 2) state-specific work
            if state is State.SEARCHING:
                if had_add or (now - last_scan_ts) >= RESCAN_INTERVAL_SEC:
                    last_scan_ts = now
                    opened = _try_open(cam_cfg, fmt_cfg)
                    if opened is not None:
                        log.info(
                            "negotiated %s %dx%d @ %.1ffps",
                            opened.negotiated.fourcc,
                            opened.negotiated.width,
                            opened.negotiated.height,
                            opened.negotiated.fps,
                        )
                        _enter_capturing(CaptureReader(opened))
                if state is State.SEARCHING:
                    shm.write(
                        black,
                        original_w=0,
                        original_h=0,
                        pad_x=0,
                        pad_y=0,
                        scale=0.0,
                        timestamp_ns=time.time_ns(),
                        connected=False,
                    )
                    sleep_for = frame_period - (time.monotonic() - now)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue

            # state == CAPTURING
            assert reader is not None
            got = reader.get()
            if got is None:
                # never produced a frame yet; small sleep then loop
                time.sleep(0.005)
                if reader.consecutive_failures > 50:  # ~2.5s of failures
                    _enter_searching("read thread reports persistent failures")
                continue
            ts, frame = got
            if ts == last_frame_ts:
                # no new frame since last write
                if now - ts > READ_TIMEOUT_SEC:
                    _enter_searching(f"no new frame for {now - ts:.2f}s")
                    continue
                time.sleep(0.001)
                continue
            last_frame_ts = ts
            lb = letterbox(frame, target_w=target_w, target_h=target_h)
            shm.write(
                lb.image,
                original_w=lb.original_w,
                original_h=lb.original_h,
                pad_x=lb.pad_x,
                pad_y=lb.pad_y,
                scale=lb.scale,
                timestamp_ns=time.time_ns(),
                connected=True,
            )
            fps_window += 1
            if args.print_fps_every > 0:
                dt = now - fps_t0
                if dt >= args.print_fps_every:
                    log.info(
                        "capture %.1f fps (%d frames in %.2fs) state=%s",
                        fps_window / dt,
                        fps_window,
                        dt,
                        state.value,
                    )
                    fps_window = 0
                    fps_t0 = now
    finally:
        if reader is not None:
            reader.stop()
        watcher.stop()
        shm.close()
        log.info("capture loop exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
