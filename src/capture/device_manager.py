"""Enumerate and select a USB camera by stable identifiers.

We never reference ``/dev/videoN`` directly because the index is assigned in
connection order. Instead we ask udev for the v4l devices, read the parent USB
device's ``ID_VENDOR_ID``/``ID_MODEL_ID`` (so VID:PID matching works even when
``/dev/v4l/by-id`` is missing), and we filter to nodes that actually capture
(a single USB camera typically exposes 2+ ``videoN`` nodes for video / metadata).
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import pyudev

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDevice:
    by_id: str | None  # basename under /dev/v4l/by-id (no path), or None if absent
    dev_path: str  # /dev/videoN
    vid: str | None  # 4-hex-digit USB vendor id, lowercase, e.g. "046d"
    pid: str | None  # 4-hex-digit USB product id, lowercase, e.g. "0892"

    @property
    def vid_pid(self) -> str | None:
        if self.vid and self.pid:
            return f"{self.vid.lower()}:{self.pid.lower()}"
        return None


def enumerate_devices(context: pyudev.Context | None = None) -> list[CameraDevice]:
    ctx = context or pyudev.Context()
    out: list[CameraDevice] = []
    for dev in ctx.list_devices(subsystem="video4linux"):
        node = dev.device_node
        if not node:
            continue
        by_id: str | None = None
        for link in dev.device_links:
            if link.startswith("/dev/v4l/by-id/"):
                by_id = Path(link).name
                break
        out.append(
            CameraDevice(
                by_id=by_id,
                dev_path=node,
                vid=(dev.get("ID_VENDOR_ID") or "").lower() or None,
                pid=(dev.get("ID_MODEL_ID") or "").lower() or None,
            )
        )
    out.sort(key=lambda d: d.dev_path)
    return out


def is_capture_capable(
    dev_path: str,
    format_hint: dict | None = None,
) -> bool:
    """Probe whether ``dev_path`` can deliver a frame.

    Some UVC cameras (e.g. Jieli USB PHY 2.0) refuse to stream at the V4L2
    driver default and only return frames once an explicit fourcc/resolution
    has been negotiated. ``format_hint`` lets callers thread the configured
    capture format through so the probe matches what the main loop will use.
    """
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return False
        if format_hint:
            from .format_negotiator import configure as _configure  # noqa: PLC0415
            try:
                _configure(
                    cap,
                    fourcc_priority=format_hint.get("fourcc_priority", ["MJPG"]),
                    width=int(format_hint.get("width", 1280)),
                    height=int(format_hint.get("height", 720)),
                    fps=int(format_hint.get("fps", 30)),
                )
            except Exception as e:  # noqa: BLE001
                log.debug("format hint apply failed on %s: %s", dev_path, e)
        ok, _ = cap.read()
        return bool(ok)
    finally:
        cap.release()


def _matches(dev: CameraDevice, pref: dict) -> bool:
    by_id_pat = pref.get("by_id")
    if by_id_pat:
        return bool(dev.by_id and fnmatch.fnmatch(dev.by_id, by_id_pat))
    vid_pid = pref.get("vid_pid")
    if vid_pid:
        return dev.vid_pid == vid_pid.lower()
    return False


def priority_rank(dev: CameraDevice, preferred: list[dict]) -> int | None:
    """Index of the first ``preferred`` entry that matches ``dev``.

    Smaller is higher priority. ``None`` means no preferred entry matches (the
    device would only be selected via the ``fallback`` policy, i.e. lowest
    priority). When a single physical camera matches several entries, the
    smallest index wins because we scan ``preferred`` in order.
    """
    for i, pref in enumerate(preferred):
        if _matches(dev, pref):
            return i
    return None


def select_device_ranked(
    devices: Iterable[CameraDevice],
    preferred: list[dict],
    fallback: str = "any",
    capture_check: bool = True,
    format_hint: dict | None = None,
    exclude_dev_paths: Iterable[str] = (),
) -> tuple[CameraDevice, int | None] | None:
    """Pick the highest-priority capture-capable device and its rank.

    Returns ``(device, rank)`` where ``rank`` is the matched ``preferred`` index
    (``None`` for a ``fallback`` match), or ``None`` if nothing is selectable.

    ``exclude_dev_paths`` skips devices already in use (e.g. the active camera
    during a CAPTURING-state re-evaluation) so the capture probe never re-opens a
    live stream. ``fallback`` is ``"any"`` to accept an unlisted camera, anything
    else (e.g. ``"none"``) to require a ``preferred`` match.
    """
    excluded = set(exclude_dev_paths)
    devs = [d for d in devices if d.dev_path not in excluded]
    for rank, pref in enumerate(preferred):
        for d in devs:
            if _matches(d, pref) and (
                not capture_check or is_capture_capable(d.dev_path, format_hint)
            ):
                return d, rank
    if fallback == "any":
        for d in devs:
            if not capture_check or is_capture_capable(d.dev_path, format_hint):
                return d, None
    return None


def select_device(
    devices: Iterable[CameraDevice],
    preferred: list[dict],
    fallback: str = "any",
    capture_check: bool = True,
    format_hint: dict | None = None,
) -> CameraDevice | None:
    """Pick the first preferred + capture-capable device, else fallback policy."""
    result = select_device_ranked(
        devices, preferred, fallback, capture_check, format_hint
    )
    return result[0] if result is not None else None
