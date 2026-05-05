"""Background YOLO inference for the WebRTC server.

Runs in a daemon thread inside the FastAPI process: reads the latest frame
from SHM, runs ``YoloWorker.predict``, and exposes the result via a
thread-safe ``get_latest()``. The async broadcast loop in ``app.py`` polls
that slot and pushes JSON to ``/ws/bbox`` clients at the camera FPS.

Why a thread, not a separate process: torch/CUDA inference releases the GIL
during the C++ kernel, so asyncio's event loop in the same process is not
blocked. Step 5 measured ~8 ms p50 inference at fp16 — well below the
33 ms 30-fps budget. If we ever see GIL contention with WebRTC encoding,
the migration path is multiprocessing.Queue between separate procs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


class YoloRunner:
    def __init__(self, shm_name: str, yolo_cfg: dict[str, Any]) -> None:
        self._shm_name = shm_name
        self._yolo_cfg = yolo_cfg
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="yolo-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            log.warning("yolo-runner thread did not exit within %.1fs", join_timeout)

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _run(self) -> None:
        # Defer heavy imports to the thread so server startup stays snappy.
        from src.capture.shm_writer import FrameSHM  # noqa: PLC0415
        from src.inference.yolo_worker import YoloWorker  # noqa: PLC0415

        log.info("yolo-runner: loading model %s (device=%s)", self._yolo_cfg["model"], self._yolo_cfg["device"])
        try:
            worker = YoloWorker(
                model_path=self._yolo_cfg["model"],
                device=self._yolo_cfg["device"],
                imgsz=self._yolo_cfg["imgsz"],
                conf=self._yolo_cfg["conf"],
                iou=self._yolo_cfg["iou"],
                half=self._yolo_cfg.get("half", False),
            )
        except Exception:
            log.exception("yolo-runner: model load failed; thread exiting")
            return

        # Wait up to 30 s for capture-run to create the SHM segment.
        shm: FrameSHM | None = None
        for _ in range(60):
            if self._stop.is_set():
                return
            try:
                shm = FrameSHM.attach(self._shm_name)
                break
            except (FileNotFoundError, RuntimeError):
                time.sleep(0.5)
        if shm is None:
            log.error("yolo-runner: SHM %r never appeared; thread exiting", self._shm_name)
            return

        log.info("yolo-runner: ready")
        self._ready.set()
        last_seq = -1
        try:
            while not self._stop.is_set():
                got = shm.read()
                if got is None:
                    time.sleep(0.005)
                    continue
                frame, meta = got
                if meta.seq == last_seq:
                    time.sleep(0.001)
                    continue
                last_seq = meta.seq
                if not meta.connected:
                    # capture is in SEARCHING; emit empty boxes so client clears overlay
                    self._publish(meta, [])
                    continue
                try:
                    dets = worker.predict(frame)
                except Exception as e:  # noqa: BLE001
                    log.warning("yolo-runner: predict raised %s", e)
                    continue
                self._publish(meta, dets)
        finally:
            shm.close()
            log.info("yolo-runner: thread exited")

    def _publish(self, meta, dets) -> None:
        boxes = [
            {
                "label": d.label,
                "conf": round(d.conf, 3),
                "x1": round(d.xyxy[0], 1),
                "y1": round(d.xyxy[1], 1),
                "x2": round(d.xyxy[2], 1),
                "y2": round(d.xyxy[3], 1),
            }
            for d in dets
        ]
        with self._lock:
            self._latest = {
                "frame_seq": meta.seq,
                "ts_ns": meta.timestamp_ns,
                "frame_w": meta.frame_w,
                "frame_h": meta.frame_h,
                "connected": meta.connected,
                "boxes": boxes,
            }
