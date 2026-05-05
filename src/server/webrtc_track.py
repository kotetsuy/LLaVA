"""SHM-backed ``VideoStreamTrack`` for aiortc.

Lazily attaches to the capture process's SHM segment on the first ``recv()``
call (so the server can start before ``capture-run``). When the SHM is not
yet present, we emit a black frame at the same target resolution; once
``capture-run`` shows up, we attach and start forwarding live frames.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from src.capture.shm_writer import FrameSHM

log = logging.getLogger(__name__)


class ShmVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(
        self,
        shm_name: str,
        frame_h: int = 720,
        frame_w: int = 1280,
        cache_ttl_sec: float = 1.0,
    ) -> None:
        super().__init__()
        self._shm_name = shm_name
        self._shm: FrameSHM | None = None
        # Black fallback at the configured target resolution. If we end up
        # attaching to an SHM with a different size, we'll regenerate.
        self._black = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        self._next_attach_attempt = 0.0
        self._attach_backoff = 1.0  # seconds between attach retries when not yet up
        # Last good frame cache — bridges transient ``shm.read()`` failures
        # (writer mid-write seqlock collisions) so the browser doesn't see
        # a one-frame black flash. Falls back to ``self._black`` once the
        # cached frame is older than ``cache_ttl_sec`` (writer probably died).
        self._last_frame: np.ndarray | None = None
        self._last_frame_seq = -1
        self._last_frame_ts = 0.0
        self._cache_ttl_sec = cache_ttl_sec

    def _try_attach(self) -> None:
        now = time.monotonic()
        if now < self._next_attach_attempt:
            return
        try:
            self._shm = FrameSHM.attach(self._shm_name)
            log.info(
                "attached SHM %s: %dx%d %s",
                self._shm_name, self._shm.frame_h, self._shm.frame_w, self._shm.pixel_format,
            )
            if (self._shm.frame_h, self._shm.frame_w) != self._black.shape[:2]:
                self._black = np.zeros(
                    (self._shm.frame_h, self._shm.frame_w, 3), dtype=np.uint8
                )
        except (FileNotFoundError, RuntimeError) as e:
            log.debug("SHM not yet available (%s); will retry", e)
            self._next_attach_attempt = now + self._attach_backoff

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        now = time.monotonic()

        if self._shm is None:
            self._try_attach()

        if self._shm is not None:
            got = self._shm.read()
            if got is not None:
                fresh, meta = got
                if meta.seq != self._last_frame_seq:
                    self._last_frame = fresh
                    self._last_frame_seq = meta.seq
                    self._last_frame_ts = now

        if self._last_frame is not None and (now - self._last_frame_ts) < self._cache_ttl_sec:
            frame_np = self._last_frame
        else:
            frame_np = self._black

        # aiortc/PyAV expects ascontiguousarray; SHM read() already returns a
        # contiguous copy, but be defensive in case upstream changes.
        if not frame_np.flags["C_CONTIGUOUS"]:
            frame_np = np.ascontiguousarray(frame_np)
        new_frame = VideoFrame.from_ndarray(frame_np, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception as e:  # noqa: BLE001
                log.warning("SHM close raised: %s", e)
            self._shm = None
        super().stop()
