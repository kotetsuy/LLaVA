"""Single-slot SharedMemory for the latest letterboxed frame.

Layout::

    offset 0   : seq_lock         uint64  (seqlock; odd=writer mid-write, even=stable)
    offset 8   : timestamp_ns     uint64
    offset 16  : original_w       uint16
    offset 18  : original_h       uint16
    offset 20  : frame_w          uint16
    offset 22  : frame_h          uint16
    offset 24  : pad_x            uint16
    offset 26  : pad_y            uint16
    offset 28  : scale            float32
    offset 32  : channels         uint8
    offset 33  : pixel_format     uint8   (0=BGR, 1=RGB)
    offset 34  : connected        uint8   (0=synthetic black "no camera", 1=live frame)
    offset 35  : _padding         1 byte
    offset 36  : frame data       (frame_w * frame_h * channels) uint8

Concurrency: a single writer (Capture proc) and N readers. The seqlock pattern
relies on hardware-atomic aligned 8-byte writes on x86_64.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Literal

import numpy as np

PixelFormat = Literal["BGR", "RGB"]
_PIXFMT_TO_INT = {"BGR": 0, "RGB": 1}
_INT_TO_PIXFMT = {v: k for k, v in _PIXFMT_TO_INT.items()}

_HEADER_FMT = "<QQHHHHHHfBBB1x"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 36


def _suppress_resource_tracker_for_shm() -> None:
    """Workaround for https://bugs.python.org/issue38119.

    The resource_tracker registers every SharedMemory the process *touches*
    (including via attach), and tries to unlink it on exit. For multi-process
    SHM where only one owner should unlink, this causes spurious warnings and
    can unlink while another process still attaches.
    Call once at the start of any process that opens a shared SHM segment.
    """
    if getattr(resource_tracker, "_llava_patched", False):
        return

    real_register = resource_tracker.register
    real_unregister = resource_tracker.unregister

    def _register(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        return real_register(name, rtype)

    def _unregister(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        return real_unregister(name, rtype)

    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    resource_tracker._CLEANUP_FUNCS.pop("shared_memory", None)
    resource_tracker._llava_patched = True


@dataclass(frozen=True)
class FrameMetadata:
    seq: int
    timestamp_ns: int
    frame_w: int
    frame_h: int
    original_w: int
    original_h: int
    pad_x: int
    pad_y: int
    scale: float
    channels: int
    pixel_format: PixelFormat
    connected: bool  # False when the writer is emitting synthetic black frames


class FrameSHM:
    """Single-slot frame SHM with seqlock concurrency."""

    def __init__(
        self,
        shm: SharedMemory,
        frame_w: int,
        frame_h: int,
        channels: int,
        pixel_format: PixelFormat,
        owns: bool,
    ) -> None:
        self._shm = shm
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.channels = channels
        self.pixel_format = pixel_format
        self._owns = owns
        self._buf = shm.buf
        self._frame_offset = _HEADER_SIZE
        self._frame_nbytes = frame_w * frame_h * channels
        # numpy view over the frame region for fast in-place copy.
        self._frame_view = np.ndarray(
            (frame_h, frame_w, channels),
            dtype=np.uint8,
            buffer=self._buf,
            offset=self._frame_offset,
        )
        self._next_seq = 0  # writer-only counter (always even after a successful write)

    @classmethod
    def create(
        cls,
        name: str,
        frame_w: int = 1280,
        frame_h: int = 720,
        channels: int = 3,
        pixel_format: PixelFormat = "BGR",
        unlink_existing: bool = True,
    ) -> "FrameSHM":
        size = _HEADER_SIZE + frame_w * frame_h * channels
        if unlink_existing:
            try:
                stale = SharedMemory(name=name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
        shm = SharedMemory(name=name, create=True, size=size)
        # zero out so an early reader sees seq == 0 (== "no data yet").
        shm.buf[:_HEADER_SIZE] = b"\x00" * _HEADER_SIZE
        return cls(shm, frame_w, frame_h, channels, pixel_format, owns=True)

    @classmethod
    def attach(cls, name: str) -> "FrameSHM":
        _suppress_resource_tracker_for_shm()
        shm = SharedMemory(name=name)
        # peek the header to recover dimensions; if seq==0 we still trust the layout.
        unpacked = struct.unpack_from(_HEADER_FMT, shm.buf, 0)
        (_, _, _, _, frame_w, frame_h, _, _, _, channels, pixfmt, _connected) = unpacked
        if frame_w == 0 or frame_h == 0 or channels == 0:
            shm.close()
            raise RuntimeError(
                f"SHM {name!r} header is empty - has the writer started yet?"
            )
        return cls(
            shm,
            frame_w=frame_w,
            frame_h=frame_h,
            channels=channels,
            pixel_format=_INT_TO_PIXFMT.get(pixfmt, "BGR"),
            owns=False,
        )

    def write(
        self,
        frame: np.ndarray,
        original_w: int,
        original_h: int,
        pad_x: int,
        pad_y: int,
        scale: float,
        timestamp_ns: int,
        connected: bool = True,
    ) -> int:
        if frame.shape != (self.frame_h, self.frame_w, self.channels):
            raise ValueError(
                f"frame shape {frame.shape} != expected "
                f"({self.frame_h},{self.frame_w},{self.channels})"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")

        # seqlock: bump to odd (writing), copy, bump to even (stable).
        odd_seq = self._next_seq + 1
        struct.pack_into("<Q", self._buf, 0, odd_seq)
        struct.pack_into(
            _HEADER_FMT,
            self._buf,
            0,
            odd_seq,
            timestamp_ns,
            original_w,
            original_h,
            self.frame_w,
            self.frame_h,
            pad_x,
            pad_y,
            float(scale),
            self.channels,
            _PIXFMT_TO_INT[self.pixel_format],
            1 if connected else 0,
        )
        np.copyto(self._frame_view, frame)
        even_seq = odd_seq + 1
        struct.pack_into("<Q", self._buf, 0, even_seq)
        self._next_seq = even_seq
        return even_seq

    def read(
        self,
        max_retries: int = 16,
        retry_sleep_sec: float = 0.0001,
    ) -> tuple[np.ndarray, FrameMetadata] | None:
        """Read latest frame, retrying through the writer's seqlock window.

        The writer's odd-seq "writing" window is ~500 µs (np.copyto on 2.6 MB).
        With the previous 8-tight-retry loop we'd give up in <10 µs, well
        before the writer finished, so reads occasionally returned ``None``
        and downstream (e.g. WebRTC track) would emit a black frame for one
        frame interval. Sleeping ~100 µs between probes lets the writer make
        progress; 16 probes covers ~1.6 ms, comfortably above the worst case.
        """
        for _ in range(max_retries):
            (s1,) = struct.unpack_from("<Q", self._buf, 0)
            if s1 == 0 or s1 & 1:
                time.sleep(retry_sleep_sec)
                continue
            unpacked = struct.unpack_from(_HEADER_FMT, self._buf, 0)
            (
                _,
                ts_ns,
                ow,
                oh,
                fw,
                fh,
                px,
                py,
                scale,
                ch,
                pixfmt_int,
                connected_int,
            ) = unpacked
            frame = np.array(self._frame_view, copy=True)
            (s2,) = struct.unpack_from("<Q", self._buf, 0)
            if s1 != s2:
                time.sleep(retry_sleep_sec)
                continue
            meta = FrameMetadata(
                seq=s1 // 2,  # human-friendly: 1, 2, 3, ...
                timestamp_ns=ts_ns,
                frame_w=fw,
                frame_h=fh,
                original_w=ow,
                original_h=oh,
                pad_x=px,
                pad_y=py,
                scale=scale,
                channels=ch,
                pixel_format=_INT_TO_PIXFMT.get(pixfmt_int, "BGR"),
                connected=bool(connected_int),
            )
            return frame, meta
        return None

    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            if self._owns:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "FrameSHM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
