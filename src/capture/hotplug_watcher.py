"""USB camera hotplug events via pyudev.

Subscribes to the ``video4linux`` subsystem and pushes ``HotplugEvent``s to a
queue that the main capture loop drains. We only forward ``add`` / ``remove``
because ``bind``/``unbind`` would double-fire for every videoN node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Literal

import pyudev

log = logging.getLogger(__name__)

Action = Literal["add", "remove"]


@dataclass(frozen=True)
class HotplugEvent:
    action: Action
    dev_path: str  # /dev/videoN
    vid: str | None
    pid: str | None
    by_id: str | None


class HotplugWatcher:
    def __init__(self) -> None:
        self._context = pyudev.Context()
        self._monitor = pyudev.Monitor.from_netlink(self._context)
        self._monitor.filter_by(subsystem="video4linux")
        self._queue: Queue[HotplugEvent] = Queue()
        self._observer = pyudev.MonitorObserver(self._monitor, self._on_event, daemon=True)

    def _on_event(self, action: str, device: pyudev.Device) -> None:
        if action not in ("add", "remove"):
            return
        node = device.device_node
        if not node:
            return
        by_id = None
        for link in device.device_links:
            if link.startswith("/dev/v4l/by-id/"):
                by_id = link.rsplit("/", 1)[-1]
                break
        ev = HotplugEvent(
            action=action,  # type: ignore[arg-type]
            dev_path=node,
            vid=(device.get("ID_VENDOR_ID") or "").lower() or None,
            pid=(device.get("ID_MODEL_ID") or "").lower() or None,
            by_id=by_id,
        )
        log.info("hotplug %s: %s (vid:pid=%s:%s by_id=%s)", ev.action, ev.dev_path, ev.vid, ev.pid, ev.by_id)
        self._queue.put(ev)

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()

    def drain(self, timeout: float | None = 0.0) -> list[HotplugEvent]:
        out: list[HotplugEvent] = []
        # Block up to ``timeout`` for the first event, then drain non-blocking.
        try:
            out.append(self._queue.get(timeout=timeout) if timeout else self._queue.get_nowait())
        except Empty:
            return out
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                return out
