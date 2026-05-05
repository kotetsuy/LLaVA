"""Tiny pub/sub for FastAPI WebSocket clients.

A single ``WsBroadcaster`` instance per topic (currently ``/ws/bbox``;
``/ws/caption`` will reuse the same class in Step 7b). Send-side errors
silently drop the offending client so a flaky tab doesn't slow others.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class WsBroadcaster:
    def __init__(self, name: str = "ws") -> None:
        self._name = name
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
        log.info("%s: client added (now %d)", self._name, len(self._clients))

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("%s: client removed (now %d)", self._name, len(self._clients))

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        text = json.dumps(message, separators=(",", ":"))
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(text)
            except Exception as e:  # noqa: BLE001
                log.debug("%s: send failed for client, dropping: %s", self._name, e)
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
            log.info("%s: dropped %d dead client(s) (now %d)", self._name, len(dead), len(self._clients))
