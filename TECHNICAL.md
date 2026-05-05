# Technical details

This document explains how the design from [`HANDOFF.md`](./HANDOFF.md) was implemented, why each design choice was made, and the gotchas discovered along the way. For setup instructions see [`README.md`](./README.md). 日本語版は [`TECHNICALJ.md`](./TECHNICALJ.md).

---

## 1. System overview

![Pipeline architecture](./docs/01_pipeline_architecture.svg)

A single NucBox EVO X2 (Ryzen AI MAX+ 395, gfx1151, 48 GB unified) runs four concurrent components and serves Chrome over WebRTC + WebSockets.

| Component | Process | Input | Output |
|---|---|---|---|
| Capture | `uv run capture-run` | USB camera | SHM (1280×720 BGR, letterboxed) |
| YOLO11m | A background thread inside `serve` | SHM | bbox JSON → `/ws/bbox` |
| VLM | `llama-server --reasoning off` (separate process) | HTTP requests (image + prompt) from `serve`'s VlmRunner | caption JSON → `/ws/caption` |
| aiortc server | `uv run serve` (FastAPI + uvicorn) | SHM | WebRTC video track + WS broadcast |

**Key design decisions**

- **SHM holds exactly one "latest frame slot"**. Capture overwrites continuously; consumers (aiortc / yolo / vlm) snapshot (`np.array(copy=True)`) at the moment they begin processing. This structurally prevents stale-frame inference backlogs.
- **Frames never travel in queues**. Only bbox / caption JSON flow via asyncio.Queue → WebSocket.
- **VLM input is JPEG-encoded bytes**. We `cv2.imencode('.jpg', ...)` the raw SHM BGR and base64-post to llama-server's `/v1/chat/completions`.
- **YOLO runs as a thread inside the `serve` process**. The original handoff suggested a separate process, but torch/CUDA releases the GIL inside its C++ kernels, so the asyncio event loop is not blocked. Step 5 measured 8 ms p50 inference at fp16 — the choice is sound.
- **VLM is the only separate process (llama-server)** because (a) we want the 21 GB GGUF resident independently of the Python venv, and (b) it lets us tune / restart llama.cpp without disturbing the main server.

---

## 2. Camera abstraction layer (CAL)

![Camera abstraction layer](./docs/02_camera_abstraction_layer.svg)

`src/capture/` collects everything from the physical layer to normalized-frame output, satisfying the requirement "must keep working when the FOV or USB port changes."

### 2.1 Device discovery (`device_manager.py`)

- `pyudev.Context().list_devices(subsystem='video4linux')` enumerates v4l devices known to udev
- USB `ID_VENDOR_ID` / `ID_MODEL_ID` are read so specs like `vid_pid="046d:0892"` work
- The stable `/dev/v4l/by-id/...` symlink (when present) becomes `by_id`
- A single USB camera often exposes multiple v4l nodes (`index0` for video, `index1` for metadata, etc.); `is_capture_capable()` filters to nodes where `cv2.VideoCapture.read()` actually succeeds
- `config.yaml`'s `camera.preferred[]` is matched (glob `usb-Logitech*` or VID:PID) → falls through to `fallback: any`

### 2.2 Format negotiation (`format_negotiator.py`)

We try the `cv2.VideoCapture.set(CAP_PROP_FOURCC, ...)` ladder in `MJPG → YUYV` priority and read back `get(CAP_PROP_*)` to see what the driver actually accepted. MJPEG is preferred because YUYV blows the USB 2.0 budget at 1280×720@30fps.

### 2.3 Two-stage letterbox (`frame_normalizer.py`)

- **CAL output is fixed at 1280×720 BGR letterbox**
- Scale = `min(target_w/src_w, target_h/src_h)`, aspect preserved on shrink, padded with `(114, 114, 114)` gray
- `pad_x / pad_y / scale / original_w / original_h` are written into the SHM header so consumers can do the inverse mapping
- **YOLO11m (640×640) and VLM (~448×448) handle the second-stage resize themselves** (Ultralytics auto-letterboxes, mtmd resizes from JPEG). CAL doesn't do a second pass — it would just duplicate work
- Bbox coordinates are absolute in the CAL-normalized frame and broadcast as-is. The Chrome `<canvas width=1280 height=720>` natural resolution is stretched by CSS to match the video — only one scale factor on the browser side

### 2.4 Hot-plug support (`hotplug_watcher.py`, `capture_session.py`)

- `pyudev.MonitorObserver(filter_by='video4linux')` puts `add` / `remove` events on a queue
- The main loop is a **two-state machine**:
  - `SEARCHING`: no working capture. Writes a black frame + `connected=False` to SHM at 30 fps so consumers immediately see "no camera"
  - `CAPTURING`: a separate `CaptureReader` daemon thread runs `cv2.VideoCapture.read()`; the main thread writes to SHM
- `CaptureReader` exists because `cv2.VideoCapture.read()` can block when the device is yanked; the daemon thread reads continuously, and `cap.release()` on stop unsticks the read
- An `add` event triggers an immediate rescan; a `remove` event for the active dev_path transitions to SEARCHING
- The aiortc `ShmVideoTrack` reads through the SHM, so swapping cameras does not break the `RTCPeerConnection` (no Chrome video drop)

---

## 3. SharedMemory design (`shm_writer.py`)

### 3.1 Layout (36 B header + frame data)

| offset | size | field |
|--------|------|-------|
| 0 | 8 | `seq_lock` (uint64): even=stable / odd=writer mid-write |
| 8 | 8 | `timestamp_ns` (uint64) |
| 16 | 2 | `original_w` (uint16) |
| 18 | 2 | `original_h` (uint16) |
| 20 | 2 | `frame_w` (uint16) |
| 22 | 2 | `frame_h` (uint16) |
| 24 | 2 | `pad_x` (uint16) |
| 26 | 2 | `pad_y` (uint16) |
| 28 | 4 | `scale` (float32) |
| 32 | 1 | `channels` (uint8) |
| 33 | 1 | `pixel_format` (uint8): 0=BGR / 1=RGB |
| 34 | 1 | `connected` (uint8): 0=synthetic black / 1=live |
| 35 | 1 | (padding) |
| 36 | W·H·3 | frame data (uint8) |

`struct` format: `<QQHHHHHHfBBB1x` (36 B confirmed via `struct.calcsize`).

### 3.2 Seqlock semantics

There is exactly one writer (Capture). On x86_64, aligned 8-byte writes are hardware-atomic, so a lock-free seqlock works:

```
Writer:
    1. seq = next odd      (= "writing" marker)
    2. write header + frame
    3. seq = next even     (= "stable" marker)

Reader:
    for retry in range(16):
        s1 = read seq
        if s1 odd: sleep(100us); continue   ← writer in flight
        copy header + frame
        s2 = read seq
        if s1 == s2: success
        else: continue                       ← writer overwrote during my copy
    return None                              ← couldn't catch a stable read in 16 tries
```

### 3.3 Fixing the "occasional black flash"

In the first version, the reader did 8 tight retries → returned `None` → `ShmVideoTrack` fell back to a black frame, producing a 1-frame black flash on Chrome. The cause:

- The writer's "odd" residency is ≈ 500 µs (`np.copyto` over 2.6 MB)
- The reader's tight 8-retry loop only spent ~8 µs total, giving up before the writer was done

Fix:

1. Insert `time.sleep(100us)` between retries in `read()`, raise the cap from 8 → 16
2. Cache "last successful frame" in `ShmVideoTrack`; on `None` reuse the cache (with a 1-second TTL guard to detect a writer that died)

After this, observed black-flash frequency is zero in normal use.

### 3.4 resource_tracker patch

`multiprocessing.shared_memory` has [bpo-38119](https://bugs.python.org/issue38119): even attaching processes try to `unlink` on exit, producing spurious warnings and double-unlinks. `_suppress_resource_tracker_for_shm()` monkey-patches `register` / `unregister` to ignore `shared_memory` — the standard workaround.

---

## 4. Inference backends

### 4.1 YOLO11m (Step 3)

| Item | Value |
|------|----|
| Backend | Ultralytics + ROCm PyTorch 2.9.1 |
| Input | 1280×720 BGR (SHM normalized frame) |
| `imgsz` | 640 (Ultralytics letterboxes internally; bboxes return in input-frame coords) |
| Quantization | fp16 (`yolo.half: true`) |
| Standalone fps | **97.8 fps** (`benchmark-yolo --source shm`, no dedup, GPU-bound) |
| Pipeline fps | **30.1 fps** (`benchmark-concurrent --no-vlm`, capped at the camera FPS) |

**Beware the baseline confusion**: `benchmark-yolo --source shm` re-reads the same SHM frame multiple times and measures pure GPU throughput (97.8 fps). `benchmark-concurrent --no-vlm` deduplicates on `meta.seq` and pins itself to the camera rate (30 fps). Step 5 must be compared against the latter; otherwise you'd misread the result as "-71% degradation."

Fallback plan (`scripts/export_yolo_onnx.py`): `model.export(format='onnx', imgsz=640, simplify=True)` writes an ONNX runnable by `onnxruntime`. CPU EP measured **15.4 fps** (insufficient as primary; this only buys graceful degradation). The MIGraphX EP requires an AMD-built ort.

### 4.2 Nemotron-3 Nano Omni (Steps 4 / 7b)

| Item | Value |
|------|----|
| Model | `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF`, Q4_K_XL (~21 GB) + mmproj-F16 (~1.5 GB) |
| Runtime | llama.cpp ROCm/HIP build (`llama-server` resident) |
| Input | 1280×720 BGR → JPEG (quality 90) → base64 → `/v1/chat/completions` |
| `n_predict` | 128 (~50 Japanese chars ≈ 60–100 tokens) |
| Standalone inference | **~1262 ms** (Step 4 mtmd-cli) / **~1300 ms** (Step 7b llama-server) |
| Concurrent inference | **~1294 ms** (Step 5, with YOLO running) — only +2.5% degradation |

**`--reasoning off` is mandatory.** Because this is a Reasoning model, the default (`auto`) lets `<think>` tags consume the entire `n_predict` budget, leaving the visible answer empty. Notably, the same GGUF behaves differently under `mtmd-cli` (which produces an empty `<think></think>` block) — this divergence between runtimes was a real source of confusion.

`VlmServerWorker` parses the llama-server `/v1/chat/completions` response by:

- Reading caption text from `choices[0].message.content`
- Stripping `<think>...</think>` defensively with a non-greedy regex
- Filling `VlmTiming` from `timings.prompt_ms / predicted_ms / *_per_second` and `usage.prompt_tokens / completion_tokens`

### 4.3 Step 5: YOLO + VLM concurrent Go/No-Go

```
                          YOLO alone   YOLO+VLM (fp32)   YOLO+VLM (fp16)
fps                       30.1         27.9              27.9
p50 latency (ms)          11.07        11.78             8.05
p99 latency (ms)          11.80        88.67             112.32
VLM median inf (ms)       —            1294              1151
VLM eval_tps              —            48.1              53.6
```

Output of `benchmark-concurrent --frames 600`. The notable finding: **fp16 widens the VLM's window**, not the YOLO fps (fps is camera-capped at 30). YOLO finishing per frame in 8 ms instead of 11 ms gives VLM bigger uninterrupted GPU windows. The p99 spikes (~100 ms) come from VLM's eval phase contending for the GPU; visually this is roughly 3 frames of bbox stutter every 2 s.

---

## 5. WebRTC delivery (`src/server/`)

### 5.1 Video track (`webrtc_track.py`)

```python
class ShmVideoTrack(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()    # aiortc paces 30 fps
        # Lazy SHM attach (works even if capture-run starts later)
        # SHM read → np.copyto → av.VideoFrame.from_ndarray(format='bgr24')
        # On read failure use last_frame cache; falls back to black after a 1 s TTL
```

- aiortc encodes on the CPU (VP8 default). 1280×720@30fps is comfortable on Ryzen AI MAX+ 395
- ROCm VCN hardware encoders are not aiortc-compatible, so we don't try
- `next_timestamp()` provides the 30 fps pacing for free

### 5.2 Signaling (`app.py`)

The `POST /offer` handler is a minimal SDP exchange (~20 lines):

```python
@app.post("/offer")
async def offer(payload: SdpPayload):
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pc.addTrack(ShmVideoTrack(...))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=payload.sdp, type=payload.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
```

Same-LAN, so no STUN/TURN — host candidates suffice. `config.yaml > server.ice_servers` is empty by default; add a STUN URL there if you need it.

### 5.3 WS broadcast (`ws_broadcaster.py`, `yolo_runner.py`, `vlm_runner.py`)

- `WsBroadcaster`: client set + asyncio Lock + JSON broadcast. Drops clients that fail to send
- `YoloRunner`: daemon thread. SHM read → predict → updates a thread-safe `_latest = {...}` slot
- `_broadcast_bbox_loop`: a 30 fps async task that dedups on `frame_seq` and pushes to `/ws/bbox`
- `VlmRunner`: same pattern with cadence 2 s. Health-checks llama-server → SHM attach → loop. Dedups on `ts_ns`
- `_broadcast_caption_loop`: 0.5 fps cadence

### 5.4 Frontend (`src/web/`)

- A 3-layer stack: `<video>` (WebRTC), `<canvas>` (bbox), semi-transparent `<div>` (caption)
- `<canvas>` has fixed `width=1280 height=720` and is positioned absolutely over the video. Bbox coords are CAL-normalized, so a single CSS scaling step is all the browser needs
- Both `overlay.js` and `caption.js` reconnect on WebSocket disconnect with exponential backoff (capped at 5 s)

---

## 6. Start / stop scripts

### `start_all.sh`

Three windows in tmux session `llava`:

1. `capture` ← `uv run capture-run`
2. `serve` ← `uv run serve` (FastAPI + YoloRunner + VlmRunner)
3. `vlm` ← `~/llama.cpp/build/bin/llama-server ... --reasoning off`

ROCm env vars are exported per pane, so even a missing `~/.bashrc` setup doesn't break the demo. After spawning, the script polls `http://localhost:8080/` with `curl` for up to 30 s before opening Chrome (or chromium / xdg-open as fallback).

### `stop_all.sh`

Sends `Ctrl-C` to each window → waits 5 s → `tmux kill-session`. Any survivor processes are caught via `pgrep -f "src.server.app|capture.main|llama-server"` and cleaned up with SIGINT → SIGKILL.

---

## 7. Gotchas discovered during implementation

### 7.1 Capture / SHM

- **Seqlock retries need a sleep.** A tight loop misses the writer's 500 µs window (§3.3)
- **`multiprocessing.shared_memory` resource_tracker patch.** Suppresses double-unlink warnings (§3.4)
- **`shm.read()` returns `(frame, meta)`, not `(ts, frame)`.** The first version of `benchmark_concurrent.py` wrote `ts, frame = got` and hit `ValueError: array truth value ambiguous`
- **`cv2.VideoCapture.read()` blocks when the USB device is yanked.** Read on a separate thread, time it out from the main thread (`CaptureReader`)

### 7.2 YOLO

- **Different baselines differ ~3×.** GPU-bound (97.8 fps) vs pipeline-bound (30.1 fps). Don't compare across them
- **fp16 helps the VLM, not the YOLO fps.** Standalone YOLO is camera-capped, but fp16 raises VLM eval_tps by +12% in the concurrent case

### 7.3 VLM (llama.cpp)

- **`llama-mtmd-cli` has no `--no-display-prompt` flag** — stdout echoes the prompt, so the Python client strips it post-hoc
- **stderr can contain non-UTF-8 bytes** (control chars from the model-load progress display). Use `subprocess.run(..., errors='replace')`
- **Naively regexing `prompt eval time` and `eval time` matches the same line twice.** A `(?<!prompt )eval time` negative lookbehind disambiguates
- **`llama-server`'s default `--reasoning auto` causes *-Reasoning models to burn `n_predict` on `<think>` tokens.** `--reasoning off` is required

### 7.4 WebRTC / frontend

- **Chrome JS cache.** When updating `caption.js`, force a hard reload (Ctrl+Shift+R)
- **POSTing the offer before ICE gathering completes loses candidates.** `await iceGatheringState === 'complete'`
- **Sign convention in metric reports.** Showing "fps decreased" as `+71.5%` is misleading. Standardize on "+ = better, - = worse"

---

## 8. Related documents

- [`HANDOFF.md`](./HANDOFF.md) — original Claude.ai design doc translated to English (the input to this implementation)
- [`README.md`](./README.md) — git clone → running, step by step
- [`docs/01_pipeline_architecture.svg`](./docs/01_pipeline_architecture.svg) and [`docs/02_camera_abstraction_layer.svg`](./docs/02_camera_abstraction_layer.svg) — design diagrams
- [`docs/LLaVA設計図.pptx`](./docs/LLaVA設計図.pptx) — original Chrome-side screen layout sketch
- 日本語版: [`HANDOFFJ.md`](./HANDOFFJ.md) / [`READMEJ.md`](./READMEJ.md) / [`TECHNICALJ.md`](./TECHNICALJ.md)
