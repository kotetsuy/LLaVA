"""Background VLM caption thread for the WebRTC server.

Mirrors ``YoloRunner`` but at the 0.5 fps caption cadence and against a
long-running ``llama-server`` process (the user starts it in T3). On loop:
read latest SHM frame → JPEG-encode → ``VlmServerWorker.predict_jpeg`` →
publish to a thread-safe slot. The async broadcast loop in ``app.py``
polls the slot and pushes caption JSON to ``/ws/caption`` clients.

Why a thread (not async): VlmServerWorker uses sync ``requests``, and a
single 1+ s blocking POST in the asyncio event loop would freeze the
WebRTC encoder. A daemon thread keeps things simple.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


class VlmRunner:
    def __init__(self, shm_name: str, vlm_cfg: dict[str, Any]) -> None:
        self._shm_name = shm_name
        self._vlm_cfg = vlm_cfg
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vlm-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            log.warning("vlm-runner thread did not exit within %.1fs", join_timeout)

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _run(self) -> None:
        import cv2  # noqa: PLC0415

        from src.capture.shm_writer import FrameSHM  # noqa: PLC0415
        from src.inference.vlm_worker import VlmServerWorker  # noqa: PLC0415

        server_cfg = self._vlm_cfg.get("server", {})
        cadence = float(server_cfg.get("cadence_sec", 2.0))
        jpeg_quality = int(server_cfg.get("jpeg_quality", 90))

        worker = VlmServerWorker(
            base_url=server_cfg.get("base_url", "http://127.0.0.1:8081"),
            prompt=self._vlm_cfg["prompt"],
            n_predict=self._vlm_cfg.get("n_predict", 96),
            temperature=self._vlm_cfg.get("temp", 0.2),
            jpeg_quality=jpeg_quality,
        )

        log.info("vlm-runner: waiting for llama-server at %s", worker.base_url)
        warned_once = False
        while not self._stop.is_set():
            if worker.health():
                break
            if not warned_once:
                log.info(
                    "vlm-runner: llama-server not yet reachable; "
                    "start it in another terminal (see README Step 7b)"
                )
                warned_once = True
            if self._stop.wait(2.0):
                return
        if self._stop.is_set():
            return
        log.info("vlm-runner: llama-server reachable")

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
            log.error("vlm-runner: SHM %r never appeared; thread exiting", self._shm_name)
            return

        log.info("vlm-runner: ready (cadence=%.1fs)", cadence)
        self._ready.set()

        next_at = time.monotonic()
        try:
            while not self._stop.is_set():
                wait = next_at - time.monotonic()
                if wait > 0 and self._stop.wait(wait):
                    return

                got = shm.read()
                if got is None:
                    next_at = time.monotonic() + cadence
                    continue
                frame, meta = got
                if not meta.connected:
                    # Camera unplugged — don't waste a VLM call on the black frame.
                    next_at = time.monotonic() + cadence
                    continue

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                )
                if not ok:
                    log.warning("vlm-runner: cv2.imencode failed; skipping frame")
                    next_at = time.monotonic() + cadence
                    continue

                t0 = time.monotonic()
                result = worker.predict_jpeg(jpeg.tobytes())
                inf_sec = time.monotonic() - t0
                if result.returncode != 0:
                    log.warning(
                        "vlm-runner: predict failed rc=%s: %s",
                        result.returncode, result.stderr[:200],
                    )
                elif not result.caption:
                    # Empty after stripping <think>...</think> usually means the model
                    # used the full n_predict budget on thinking. See README Step 7b
                    # for --reasoning off.
                    log.warning(
                        "vlm-runner: empty caption after strip; raw=%r (consider --reasoning off)",
                        result.stdout[:300],
                    )
                    self._publish(meta, result, inf_sec)
                else:
                    log.info(
                        "vlm-runner: caption (%.0fms) %r",
                        inf_sec * 1000, result.caption[:80],
                    )
                    self._publish(meta, result, inf_sec)

                next_at = time.monotonic() + cadence
        finally:
            shm.close()
            log.info("vlm-runner: thread exited")

    def _publish(self, meta, result, inf_sec: float) -> None:
        t = result.timing
        timing = {
            "inference_ms": round(
                t.inference_ms if t.inference_ms == t.inference_ms else inf_sec * 1000,
                1,
            ),
            "prompt_eval_ms": (
                round(t.prompt_eval_ms, 1) if t.prompt_eval_ms == t.prompt_eval_ms else None
            ),
            "eval_ms": round(t.eval_ms, 1) if t.eval_ms == t.eval_ms else None,
            "n_prompt_tokens": t.n_prompt_tokens or None,
            "n_eval_tokens": t.n_eval_tokens or None,
            "eval_tps": round(t.eval_tps, 1) if t.eval_tps == t.eval_tps else None,
        }
        with self._lock:
            self._latest = {
                "ts_ns": meta.timestamp_ns,
                "frame_seq": meta.seq,
                "caption": result.caption,
                "timing": timing,
            }
