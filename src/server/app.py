"""FastAPI + aiortc signaling server.

Step 6: serves ``index.html`` and exchanges SDP at ``POST /offer``.
Step 7a: adds an in-process YOLO inference thread + ``/ws/bbox`` broadcast.
Step 7b (planned): ``/ws/caption`` driven by VLM (llama-server transition).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .vlm_runner import VlmRunner
from .webrtc_track import ShmVideoTrack
from .ws_broadcaster import WsBroadcaster
from .yolo_runner import YoloRunner

log = logging.getLogger("server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CONFIG_PATH = Path(os.environ.get("LLAVA_CONFIG", "config.yaml"))

_pcs: set[RTCPeerConnection] = set()
_bbox_ws = WsBroadcaster(name="ws/bbox")
_caption_ws = WsBroadcaster(name="ws/caption")
_yolo_runner: YoloRunner | None = None
_vlm_runner: VlmRunner | None = None


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


async def _broadcast_bbox_loop(period: float = 1.0 / 30) -> None:
    """Push the latest detections to /ws/bbox subscribers at ~30 fps.

    Skips when no clients are connected (so YoloRunner still runs but we
    don't waste JSON encoding) and dedups on ``frame_seq`` so we never
    emit the same detection twice.
    """
    last_seq = -1
    try:
        while True:
            await asyncio.sleep(period)
            if _bbox_ws.n_clients == 0:
                continue
            if _yolo_runner is None:
                continue
            d = _yolo_runner.get_latest()
            if d is None or d["frame_seq"] == last_seq:
                continue
            last_seq = d["frame_seq"]
            await _bbox_ws.broadcast(d)
    except asyncio.CancelledError:
        return


async def _broadcast_caption_loop(period: float = 0.5) -> None:
    """Push fresh captions to /ws/caption subscribers, polling at 2 Hz.

    VLM produces ~0.5 fps; we poll faster than that so a new caption gets
    out within ~500 ms of being computed. Dedups on ``ts_ns``.
    """
    last_ts = -1
    try:
        while True:
            await asyncio.sleep(period)
            if _caption_ws.n_clients == 0:
                continue
            if _vlm_runner is None:
                continue
            d = _vlm_runner.get_latest()
            if d is None or d["ts_ns"] == last_ts:
                continue
            last_ts = d["ts_ns"]
            await _caption_ws.broadcast(d)
    except asyncio.CancelledError:
        return


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _yolo_runner, _vlm_runner
    log.info("server starting (config=%s, web=%s)", CONFIG_PATH, WEB_DIR)

    cfg = _load_config()
    _yolo_runner = YoloRunner(shm_name=cfg["shm"]["name"], yolo_cfg=cfg["yolo"])
    _yolo_runner.start()
    _vlm_runner = VlmRunner(shm_name=cfg["shm"]["name"], vlm_cfg=cfg["vlm"])
    _vlm_runner.start()

    bbox_task = asyncio.create_task(_broadcast_bbox_loop(), name="bbox-broadcast")
    caption_task = asyncio.create_task(_broadcast_caption_loop(), name="caption-broadcast")

    try:
        yield
    finally:
        log.info("server shutting down; closing %d peer connection(s)", len(_pcs))
        for task in (bbox_task, caption_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(*(pc.close() for pc in _pcs), return_exceptions=True)
        _pcs.clear()
        if _yolo_runner is not None:
            _yolo_runner.stop()
            _yolo_runner = None
        if _vlm_runner is not None:
            _vlm_runner.stop()
            _vlm_runner = None


app = FastAPI(lifespan=lifespan, title="LLaVA WebRTC demo")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class SdpPayload(BaseModel):
    sdp: str
    type: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/offer")
async def offer(payload: SdpPayload) -> JSONResponse:
    cfg = _load_config()

    ice_servers = [RTCIceServer(urls=u) for u in cfg.get("server", {}).get("ice_servers", [])]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    _pcs.add(pc)
    log.info("new peer connection (now %d total)", len(_pcs))

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        log.info("connection state -> %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            _pcs.discard(pc)

    try:
        target = cfg["camera"]["output"]["target"]
        track = ShmVideoTrack(
            shm_name=cfg["shm"]["name"], frame_h=target[1], frame_w=target[0]
        )
        pc.addTrack(track)

        sdp_offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
        await pc.setRemoteDescription(sdp_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception:
        await pc.close()
        _pcs.discard(pc)
        raise

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


@app.websocket("/ws/bbox")
async def ws_bbox(ws: WebSocket) -> None:
    await ws.accept()
    await _bbox_ws.add(ws)
    try:
        # No client-to-server messages expected; receive_text() will raise
        # on disconnect, ending the loop cleanly.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _bbox_ws.remove(ws)


@app.websocket("/ws/caption")
async def ws_caption(ws: WebSocket) -> None:
    await ws.accept()
    await _caption_ws.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _caption_ws.remove(ws)


def run() -> None:
    """Entry point for ``uv run serve``."""
    import uvicorn  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = _load_config()
    sc = cfg.get("server", {})
    host = sc.get("host", "0.0.0.0")
    port = int(sc.get("port", 8080))
    log.info("starting uvicorn on %s:%d", host, port)
    uvicorn.run("src.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
