# USB camera → VLM/YOLO → Chrome streaming demo: design document

## 0. About this document

A handoff from the design discussion done in Claude.ai (web) to Claude Code on Linux (NucBox EVO X2). Read this together with `LLaVA設計図.pptx` (the original screen-layout sketch) and the two SVG diagrams in the same folder.

- `LLaVA設計図.pptx` — Chrome screen layout sketch (USB camera + "what is this?" window + YOLO bbox)
- `01_pipeline_architecture.svg` — process split and IPC architecture
- `02_camera_abstraction_layer.svg` — camera abstraction layer with the two-stage letterbox

### Design ↔ implementation delta (added post-implementation)

This document preserves the original Claude.ai design discussion verbatim. The notable change that emerged during implementation:

- **The video transport is MJPEG (`multipart/x-mixed-replace`) over HTTP, not WebRTC (aiortc)**. On a fully offline LAN (Wi-Fi off, no Internet) Chrome refuses to emit even a single ICE host candidate, leaving the connection wedged in `connecting`. Since we cannot influence Chrome's gathering behavior we switched to a transport that needs no ICE at all. See §6 "Implementation note" at the bottom and `TECHNICAL.md` §5 for the full story.
- Everything else (Capture / CAL / SHM / YOLO / VLM / WS bbox & caption) is implemented as described in this document.

## 1. What we want to build

A demo that runs the following concurrently on the NucBox EVO X2 and shows the result with low latency in Chrome:

1. Capture from a USB camera (30 fps)
2. Stream the video to Chrome with low latency over WebRTC
3. Run YOLO11m object detection and overlay bboxes at 30 fps
4. Once every 2 seconds, ask Nemotron Nano Omni "what do you see?" in Japanese and show the answer in a dedicated window in Chrome

See `LLaVA設計図.pptx` for the screen mock.

## 2. Target environment (assumptions)

| Item | Value |
|------|------|
| Machine | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151 / RDNA 3.5) |
| Memory | 48 GB unified (BIOS-allocated VRAM) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.2 (`/opt/rocm` symlink) |
| Env vars | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |
| Existing assets | llama.cpp ROCm/HIP build (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`) |
| Existing assets | ROCm PyTorch wheels (from repo.radeon.com) |
| Existing assets | Practical experience with WhisperX / Ollama / VOICEVOX local AI pipelines |

## 3. Architecture overview

See `01_pipeline_architecture.svg`. We split work across 4 processes and decouple them with SharedMemory + `asyncio.Queue`.

| Process | Role | Implementation strategy |
|---------|------|----------|
| Capture | Pull frames from the USB camera, write to SHM | OpenCV + v4l2 + pyudev |
| YOLO11m | 30 fps object detection, push bboxes to a queue | Ultralytics + ROCm PyTorch |
| VLM | 0.5 fps (one inference / 2 s) Japanese caption generation | llama.cpp (ROCm/HIP) + Nemotron Nano Omni |
| aiortc server | WebRTC video + WebSocket bbox/caption broadcast | aiortc + FastAPI |

### IPC design highlights

- **SHM holds exactly one "latest frame slot"**: Capture always overwrites; YOLO/VLM take a snapshot (`numpy.copy()`) at the moment they begin inference.
- **Queues carry results only** (no frames): prevents inference backlogs of stale frames.
- **VLM receives JPEG-encoded bytes**: llama.cpp's mtmd accepts file or bytes input, so we `cv2.imencode('.jpg', ...)` the SHM raw frame before handing it over.

## 4. Camera abstraction layer (CAL) — important

See `02_camera_abstraction_layer.svg`. The requirement "must keep working when the camera FOV changes or the USB port changes" demands the following.

### 4.1 Dynamic device discovery

- The numbering of `/dev/video*` shifts with connection order, so **never reference it directly**
- Identify cameras by udev's `/dev/v4l/by-id/usb-...` path or by VID/PID
- At startup, run something equivalent to `v4l2-ctl --list-devices` (in Python, `v4l2-python3` or `subprocess` + `v4l2-ctl`)
- A single USB camera typically exposes multiple `/dev/videoN` nodes; use `VIDIOC_QUERYCAP` to filter to the ones that actually capture

### 4.2 Format negotiation

- `v4l2-ctl --list-formats-ext` to enumerate supported resolutions / FPS / FourCC
- Pick the best match for the desired (1280×720@30fps)
- **Prefer MJPEG** — its USB-bandwidth efficiency beats YUYV by a wide margin
- Fallback order: MJPG → YUYV

### 4.3 Two-stage letterbox

This is the most important part of the new requirements.

- **CAL output = fixed 1280×720 letterboxed RGB**: matches the aiortc transmit resolution
- **YOLO11m input = 640×640**: Ultralytics' `model.predict(imgsz=640)` does the letterbox + inverse mapping internally (no manual code needed)
- **VLM input ≈ 448×448**: simple resize (the letterbox is already done in CAL) — sized per model spec
- **Bbox coordinates are absolute coords in the SHM-normalized frame**, sent as-is over the WS. The Chrome side only needs to handle one scale factor

### 4.4 Hot-plug support

- `pyudev.Monitor` subscribed to `subsystem='video4linux'`
- On `remove`: safely close `cv2.VideoCapture`, generate a black frame + a "no camera" caption
- On `add`: re-pick by policy and reopen
- **Keep the aiortc VideoTrack alive — only swap the inner source**, so the Chrome connection doesn't drop
- `cv2.VideoCapture.read()` can block when the device is yanked → run it in a separate thread with a 500 ms timeout

### 4.5 Example config

```yaml
# config.yaml
camera:
  preferred:
    - by_id: usb-Logitech_HD_Pro_Webcam_C920*
    - vid_pid: "046d:0892"
  fallback: any
  format:
    width: 1280
    height: 720
    fps: 30
    fourcc_priority: [MJPG, YUYV]
  output:
    target: [1280, 720]
    keep_aspect: true
    fisheye_correct: false  # fisheye correction is opt-in only (per-model calibration required)
```

## 5. Inference backend choice (decided)

| Use | Backend | Notes |
|------|------------|------|
| Nemotron Nano Omni | llama.cpp (ROCm/HIP) | Existing build assets; mtmd for image input |
| YOLO11m | ROCm PyTorch (Ultralytics) | Has a fallback plan (below) |
| Video stream | WebRTC (aiortc) | Low-latency priority |

### Fallback plan: move YOLO to CPU/iGPU

Co-locating llama.cpp and PyTorch ROCm on the same GPU may cause both to slow down due to HIP-context-switch costs. By exporting via Ultralytics' `model.export(format='onnx')` we can switch to ONNX Runtime CPU/MIGraphX. **Set up the ONNX export from the start.**

## 6. WebRTC delivery realities

- Subclass aiortc's `VideoStreamTrack`, return the latest frame from `recv()`
- Encoder is VP8 (default) or H.264 — **CPU encoding only** (aiortc does not support ROCm VCN)
- 1280×720@30fps VP8 is comfortable on Ryzen AI MAX+ 395
- Signaling: SDP offer/answer exchange over FastAPI (~20 lines)
- On the same LAN, no STUN/TURN needed — host candidates are enough

### Chrome side

- `<video>`: receives the WebRTC track, displays raw video only
- `<canvas>` overlay: receives bbox over WS, draws at 30 fps
- `<div>`: receives caption over WS, displays at 0.5 fps
- "Always draw the latest bbox" sync is fine. For strict synchronization use `requestVideoFrameCallback` + PTS alignment

### Implementation note: WebRTC → MJPEG

Once implemented and run on an offline LAN we observed that Chrome does not emit a single ICE host candidate (`chrome://webrtc-internals` never fires `onicecandidate`; `iceState` stays at `new`). With Wi-Fi off and no non-loopback interface, Chrome's privacy logic abandons gathering. Patching aioice's loopback filter on the server side, adding a dummy STUN entry, and even implementing trickle ICE all proved useless because Chrome never produces a local candidate to pair with.

We therefore replaced WebRTC with MJPEG over HTTP (`multipart/x-mixed-replace; boundary=frame`):

- Server: `GET /stream.mjpg` reads SHM, runs `cv2.imencode('.jpg', ...)` off-thread, and yields each frame as a multipart chunk
- Chrome: `<video>` becomes `<img src="/stream.mjpg">` with `onload` / `onerror` for a simple exponential-backoff reconnect

No ICE, STUN, TURN, aiortc, or `RTCPeerConnection` — just plain HTTP, which works both on the LAN and offline. The latency penalty over WebRTC is real but acceptable for this demo. Details in `TECHNICAL.md` §5.

## 7. Recommended build order (important)

Wiring everything at once makes debugging hard, so bring it up incrementally. **Step 4 is the Go/No-Go.**

1. **Capture + SHM**: confirm a separate process can read the SHM
2. **CAL standalone**: swap different cameras and confirm the letterboxed output stays stable
3. **YOLO11m standalone**: measure whether ROCm PyTorch hits 30 fps
4. **Nemotron Nano Omni standalone**: measure one-image inference time in llama.cpp (within the 2-second budget?)
5. **YOLO + VLM concurrent**: measure tok/s and fps degradation ← **Go/No-Go**
6. **aiortc server**: add last; confirm reception in Chrome
7. **Hot-plug**: add pyudev monitoring last

## 8. Proposed project structure

```
~/projects/webcam-vlm-yolo/
├── README.md
├── config.yaml
├── pyproject.toml          # uv or poetry
├── docs/
│   ├── 01_pipeline_architecture.svg
│   ├── 02_camera_abstraction_layer.svg
│   └── LLaVA設計図.pptx
├── src/
│   ├── capture/
│   │   ├── device_manager.py      # v4l2 enumeration, by-id selection
│   │   ├── format_negotiator.py   # resolution / FPS / FourCC
│   │   ├── frame_normalizer.py    # letterbox + metadata
│   │   ├── hotplug_watcher.py     # pyudev monitor
│   │   └── shm_writer.py          # SharedMemory writes
│   ├── inference/
│   │   ├── yolo_worker.py         # YOLO11m proc
│   │   └── vlm_worker.py          # llama.cpp invocation
│   ├── server/
│   │   ├── app.py                 # FastAPI + aiortc
│   │   ├── webrtc_track.py        # VideoStreamTrack
│   │   └── ws_broadcaster.py      # bbox/caption push
│   └── web/
│       ├── index.html
│       ├── overlay.js
│       └── style.css
└── scripts/
    ├── benchmark_yolo.py
    ├── benchmark_vlm.py
    └── list_cameras.py
```

## 9. Open questions (confirm / discuss in Claude Code)

- Exact model name and GGUF source for Nemotron Nano Omni (and llama.cpp support status)
- VLM vision-encoder input size (448, 384, or other)
- Whether MJPEG 30 fps is stable on actual USB cameras
- Whether strict sync via Chrome's `requestVideoFrameCallback` is required
