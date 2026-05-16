"""FastAPI server: SHM → MJPEG over HTTP, plus YOLO/VLM WebSocket channels.

WebRTC was the original transport (see ``webrtc_track.py``) but it fails in
fully offline LANs because Chrome refuses to gather even a loopback ICE
candidate when no internet-reachable interface is present. We fall back to
``multipart/x-mixed-replace`` MJPEG: plain HTTP, no ICE, works as long as
the page can be fetched.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.capture.shm_writer import FrameSHM

from .vlm_runner import VlmRunner
from .ws_broadcaster import WsBroadcaster
from .yolo_runner import YoloRunner

log = logging.getLogger("server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CONFIG_PATH = Path(os.environ.get("LLAVA_CONFIG", "config.yaml"))

_bbox_ws = WsBroadcaster(name="ws/bbox")
_caption_ws = WsBroadcaster(name="ws/caption")
_yolo_runner: YoloRunner | None = None
_vlm_runner: VlmRunner | None = None


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


async def _broadcast_bbox_loop(period: float = 1.0 / 30) -> None:
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
        log.info("server shutting down")
        for task in (bbox_task, caption_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if _yolo_runner is not None:
            _yolo_runner.stop()
            _yolo_runner = None
        if _vlm_runner is not None:
            _vlm_runner.stop()
            _vlm_runner = None


app = FastAPI(lifespan=lifespan, title="LLaVA MJPEG demo")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


_MJPEG_BOUNDARY = "frame"


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    """Multipart MJPEG stream sourced from the capture-run SHM frame.

    Works offline (plain HTTP, no ICE) — usable directly as
    ``<img src="/stream.mjpg">``.
    """
    cfg = _load_config()
    shm_name = cfg["shm"]["name"]
    fps_cap = float(cfg["camera"]["format"].get("fps", 30))
    frame_period = 1.0 / fps_cap if fps_cap > 0 else 1.0 / 30
    jpeg_quality = int(cfg.get("server", {}).get("mjpeg_quality", 80))
    target_w, target_h = cfg["camera"]["output"]["target"]
    boundary = _MJPEG_BOUNDARY

    async def gen():
        shm: FrameSHM | None = None
        last_seq = -1
        black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        next_attach_attempt = 0.0
        try:
            while True:
                t_start = time.monotonic()

                if shm is None and t_start >= next_attach_attempt:
                    try:
                        shm = FrameSHM.attach(shm_name)
                        log.info("mjpeg: attached SHM %s", shm_name)
                    except (FileNotFoundError, RuntimeError) as e:
                        log.debug("mjpeg: SHM not yet available (%s)", e)
                        next_attach_attempt = t_start + 1.0

                frame = black
                if shm is not None:
                    got = shm.read()
                    if got is not None:
                        fresh, meta = got
                        if meta.seq != last_seq:
                            frame = fresh
                            last_seq = meta.seq
                        else:
                            frame = fresh  # re-emit last good frame to keep stream alive

                ok, jpeg = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if ok:
                    jpeg_bytes = jpeg.tobytes()
                    yield (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                    ).encode("ascii") + jpeg_bytes + b"\r\n"

                elapsed = time.monotonic() - t_start
                sleep_for = max(0.0, frame_period - elapsed)
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            raise
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception as e:  # noqa: BLE001
                    log.warning("mjpeg: shm.close() raised %s", e)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws/bbox")
async def ws_bbox(ws: WebSocket) -> None:
    await ws.accept()
    await _bbox_ws.add(ws)
    try:
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
