"""Camera-open helper + threaded reader with timeout.

``cv2.VideoCapture.read()`` can block if the USB device is yanked mid-grab. We
run reads in a daemon thread, expose the most-recent frame to the main loop,
and let the main loop time out (e.g. 500 ms) to drive a SEARCHING transition
without waiting on the syscall.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .device_manager import CameraDevice
from .format_negotiator import NegotiatedFormat, configure as configure_format

log = logging.getLogger(__name__)


@dataclass
class OpenedCamera:
    device: CameraDevice
    cap: cv2.VideoCapture
    negotiated: NegotiatedFormat


def open_camera(
    device: CameraDevice,
    fourcc_priority: list[str],
    width: int,
    height: int,
    fps: int,
) -> OpenedCamera | None:
    cap = cv2.VideoCapture(device.dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        log.warning("cv2 failed to open %s", device.dev_path)
        cap.release()
        return None
    fmt = configure_format(cap, fourcc_priority, width, height, fps)
    log.info(
        "opened %s -> %s %dx%d @ %.1ffps",
        device.dev_path,
        fmt.fourcc,
        fmt.width,
        fmt.height,
        fmt.fps,
    )
    # Sanity probe: a single successful read confirms the pipeline is alive
    # before we hand the cap to the read thread.
    ok, _ = cap.read()
    if not ok:
        log.warning("first read() on %s failed; aborting open", device.dev_path)
        cap.release()
        return None
    return OpenedCamera(device=device, cap=cap, negotiated=fmt)


class CaptureReader:
    """Background thread that keeps the latest frame from a ``cv2.VideoCapture``."""

    def __init__(self, opened: OpenedCamera) -> None:
        self._opened = opened
        self._cap = opened.cap
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        self._consec_failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture-reader", daemon=True)

    @property
    def device(self) -> CameraDevice:
        return self._opened.device

    @property
    def negotiated(self) -> NegotiatedFormat:
        return self._opened.negotiated

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except cv2.error as e:
                log.warning("cap.read() raised %s", e)
                ok, frame = False, None
            if not ok or frame is None:
                self._consec_failures += 1
                # Don't tight-loop on failures; the device may have just been removed.
                time.sleep(0.05)
                continue
            self._consec_failures = 0
            with self._lock:
                self._latest = (time.monotonic(), frame)

    def get(self) -> tuple[float, np.ndarray] | None:
        """Return ``(ts_monotonic, frame)`` of the most recent frame, or None."""
        with self._lock:
            return self._latest

    @property
    def consecutive_failures(self) -> int:
        return self._consec_failures

    def stop(self) -> None:
        self._stop.set()
        # Releasing the cap unblocks a stuck read() on Linux V4L2.
        try:
            self._cap.release()
        except Exception as e:  # noqa: BLE001
            log.warning("cap.release() raised %s", e)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.warning("capture-reader thread did not exit within 2s; leaking as daemon")
