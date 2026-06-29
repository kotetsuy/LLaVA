# Technical details

This document explains how the design from [`HANDOFF.md`](./HANDOFF.md) was implemented, why each design choice was made, and the gotchas discovered along the way. For setup instructions see [`README.md`](./README.md). 日本語版は [`TECHNICALJ.md`](./TECHNICALJ.md).

> **Note on transport**: the original design used WebRTC (aiortc) for video, but on a fully offline LAN (Wi-Fi off / no Internet) Chrome refuses to emit a single ICE host candidate and the connection wedges. We migrated to MJPEG (`multipart/x-mixed-replace`) over HTTP. See §5 and §7.4 for the full story.

---

## 1. System overview

![Pipeline architecture](./docs/01_pipeline_architecture.svg)

A single NucBox EVO X2 (Ryzen AI MAX+ 395, gfx1150, 48 GB unified) runs four concurrent components and serves Chrome over MJPEG (`/stream.mjpg`) + WebSockets.

| Component | Process | Input | Output |
|---|---|---|---|
| Capture | `uv run capture-run` | USB camera | SHM (1280×720 BGR, letterboxed) |
| YOLO11m | A background thread inside `serve` | SHM | bbox JSON → `/ws/bbox` |
| VLM | `llama-server --reasoning off` (separate process) | HTTP requests (image + prompt) from `serve`'s VlmRunner | caption JSON → `/ws/caption` |
| MJPEG server | `uv run serve` (FastAPI + uvicorn) | SHM | `/stream.mjpg` (multipart/x-mixed-replace) + WS broadcast |

**Key design decisions**

- **SHM holds exactly one "latest frame slot"**. Capture overwrites continuously; consumers (MJPEG / yolo / vlm) snapshot (`np.array(copy=True)`) at the moment they begin processing. This structurally prevents stale-frame inference backlogs.
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
- `config.yaml`'s `camera.preferred[]` is an **ordered priority list** of cameras (each entry one camera via `by_id` glob or `vid_pid`). `select_device_ranked()` returns the highest-priority capture-capable match plus its rank (the `preferred` index; `None` = matched only via `fallback`). `fallback: any` accepts an unlisted camera; `fallback: none` requires a `preferred` match. So with several listed cameras connected, only the top-ranked one is shown

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
- A `remove` event for the active dev_path transitions to SEARCHING, which rescans and reconnects to any remaining listed camera
- An `add` event in SEARCHING triggers an immediate rescan. An `add` in CAPTURING arms a `preempt_settle_sec` debounce timer (udev fires before the device is ready); when it expires, the loop re-evaluates excluding the active dev_path and, **only if a strictly higher-priority camera appeared**, opens it first and then swaps the `CaptureReader` (so no black frame is inserted). A lower- or equal-priority `add` is ignored — that is what keeps "show only one when several are plugged." Set `preempt: false` to disable live switching
- The MJPEG stream reads through the SHM, so swapping cameras does not break the `<img src="/stream.mjpg">` connection (the HTTP stream stays alive, with black frames bridging the gap until the live feed returns)

### 2.5 Adding a camera

Enabling a new USB camera is just one entry in `config.yaml`'s `camera.preferred[]` — no code changes.

1. **Find its identifier** — plug the camera in and run `uv run list-cameras`. Rows with `CAPTURE = yes` are the nodes that can deliver frames (one camera exposes several nodes: `index0` for video, `index1` for metadata, etc.). Note that row's `BY-ID` (= `by_id`) or `VID:PID` (= `vid_pid`).

   ```
   CAPTURE  DEV            VID:PID     BY-ID
   yes      /dev/video0    056e:701a   usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index0
   no       /dev/video1    056e:701a   usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index1
   ```

2. **Pick the identifier** — use `by_id` when you need to distinguish individual units of the same model (unique if the `BY-ID` carries a serial; the trailing part can be a `*` glob), or `vid_pid` for a quick model-level match (compared lowercased, exact). Write **only one** of them per entry.

3. **Append it to `camera.preferred[]`** — the list is **highest priority first**. With several listed cameras connected, only the top-ranked match is shown. Put it first to make it the top choice, or last for fallback treatment (used only when nothing higher matches). `name` is a free-form label for logs and is not used by the selection logic.

   ```yaml
   camera:
     preferred:
       - name: 2k-usb-camera          # rank 0 (top priority)
         by_id: usb-DC474C08_..._2K_USB_Camera_...*
       - name: elecom-2mp             # last = fallback treatment
         by_id: usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index0
   ```

4. **Apply it** — restart capture to reload the edited `config.yaml` (`./stop_all.sh && ./start_all.sh`). With `preempt: true`, hot-plugging a higher-priority camera while running also switches over automatically, but the config edit itself takes effect on restart.

> With `fallback: any` (the default), a camera not listed in `preferred` still connects at the lowest priority. Set `fallback: none` to use **only** the cameras listed in `preferred`.

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

In the first version, the reader did 8 tight retries → returned `None` → the downstream consumer (then `ShmVideoTrack`, now the MJPEG generator) fell back to a black frame, producing a 1-frame black flash on Chrome. The cause:

- The writer's "odd" residency is ≈ 500 µs (`np.copyto` over 2.6 MB)
- The reader's tight 8-retry loop only spent ~8 µs total, giving up before the writer was done

Fix:

1. Insert `time.sleep(100us)` between retries in `read()`, raise the cap from 8 → 16
2. Cache "last successful frame" downstream; on `None`, reuse the cache (with a 1-second TTL guard to detect a writer that died) — first added in `ShmVideoTrack`, mirrored in the current MJPEG generator

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

## 5. Video delivery (`src/server/`)

### 5.1 Why we dropped WebRTC

The original implementation used aiortc + `RTCPeerConnection` to serve a WebRTC video track. On the same LAN we assumed host candidates alone would be enough, no STUN/TURN required. Once the demo ran on an offline LAN (Wi-Fi off, no Internet), we hit a dead end:

- `POST /offer` returned 200
- aiortc transitioned to `connection state -> connecting`
- …and never advanced — neither `connected` nor `failed`. The browser stayed black

`chrome://webrtc-internals` showed that Chrome's `onicecandidate` **never fired**, and `iceState` remained `new`. Without a non-loopback interface, Chrome's WebRTC stack stops emitting any local host candidates at all. Things we tried and the outcomes:

1. **Monkey-patch aioice's loopback filter** so the server publishes `127.0.0.1` as a host candidate → the server side now offered a usable candidate, but the browser still produced none, so no pair could form
2. **Switch the page URL between `http://localhost:8080/` and `http://127.0.0.1:8080/`** → no change
3. **Disable `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns`** → no change
4. **Pass an unreachable dummy STUN to `RTCPeerConnection({iceServers: [{urls: 'stun:127.0.0.1:3478'}]})`** → no change
5. **Implement trickle ICE** (server `POST /candidate` + client `icecandidate` posting) → Chrome doesn't fire the event in the first place, so trickle has nothing to send

Since we couldn't change Chrome's behavior, we switched to a transport that doesn't need ICE at all.

### 5.2 MJPEG stream (`app.py`)

`GET /stream.mjpg` returns a `StreamingResponse` with `multipart/x-mixed-replace; boundary=frame`:

```python
@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    async def gen():
        shm: FrameSHM | None = None
        last_seq = -1
        black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        while True:
            t_start = time.monotonic()
            if shm is None:                              # lazy attach
                try: shm = FrameSHM.attach(shm_name)
                except (FileNotFoundError, RuntimeError): pass
            frame = black
            if shm is not None and (got := shm.read()) is not None:
                fresh, meta = got
                if meta.seq != last_seq:
                    frame = fresh; last_seq = meta.seq
                else:
                    frame = fresh
            ok, jpeg = await asyncio.to_thread(
                cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                yield (f"--frame\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(max(0.0, frame_period - (time.monotonic() - t_start)))
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", ...)
```

Highlights:

- **Lazy attach** — capture-run can start after the server; until SHM exists we emit `black` to keep the HTTP stream alive
- **JPEG encoding off-thread** — `cv2.imencode` is CPU-heavy, so we hand it to `asyncio.to_thread` (Python 3.9+) to keep the event loop responsive
- **Rate limit** — `config.yaml > camera.format.fps` (= 30) becomes `frame_period`, enforced via `asyncio.sleep`
- **Quality** — `config.yaml > server.mjpeg_quality` (default 80). At 1280×720 a single frame is ~80–120 KB, i.e. ~20–30 Mbps at 30 fps
- **No ICE / STUN / TURN** — plain HTTP, so the LAN and offline cases behave the same way

### 5.3 WS broadcast (`ws_broadcaster.py`, `yolo_runner.py`, `vlm_runner.py`)

The WebRTC → MJPEG migration left these untouched:

- `WsBroadcaster`: client set + asyncio Lock + JSON broadcast. Drops clients that fail to send
- `YoloRunner`: daemon thread. SHM read → predict → updates a thread-safe `_latest = {...}` slot
- `_broadcast_bbox_loop`: a 30 fps async task that dedups on `frame_seq` and pushes to `/ws/bbox`
- `VlmRunner`: same pattern with cadence 2 s. Health-checks llama-server → SHM attach → loop. Dedups on `ts_ns`
- `_broadcast_caption_loop`: 0.5 fps cadence

### 5.4 Frontend (`src/web/`)

- A 3-layer stack: `<img id="stream" src="/stream.mjpg">` (MJPEG), `<canvas>` (bbox), semi-transparent `<div>` (caption)
- `<canvas>` has fixed `width=1280 height=720` and is positioned absolutely over the `<img>`. Bbox coords are CAL-normalized, so a single CSS scaling step is all the browser needs
- `<img>` sets the status row to `streaming` on `load`; on `error` it reconnects with exponential backoff (capped at 5 s) and a `?t=<ts>` cache-buster on the `src`
- Both `overlay.js` and `caption.js` reconnect on WebSocket disconnect with exponential backoff (capped at 5 s)

### 5.5 What we left in the tree

`src/server/webrtc_track.py` is no longer imported but remains in the repo, in case a future deployment can provide a STUN/TURN server and we want to revisit WebRTC. `pyproject.toml`'s `[webrtc]` extra still installs `aiortc` for the same reason — the current `serve` simply doesn't import it.

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

### 7.4 Video transport / frontend

- **Chrome won't emit any ICE candidates when fully offline.** Wi-Fi off + only loopback → host candidate gathering is silently abandoned (`chrome://webrtc-internals` never fires `onicecandidate`, `iceState` stuck at `new`). Patching aioice's loopback exclusion or implementing trickle ICE does not help, because the browser itself produces nothing to pair against. We migrated to MJPEG specifically to dodge this — see §5.1
- **MJPEG bandwidth is ~20–30 Mbps per client.** At 1280×720 / 30 fps / JPEG quality 80 a frame is ~80–120 KB. Bandwidth and `cv2.imencode` CPU both scale linearly with the number of connected browsers
- **Chrome JS cache.** When `caption.js` or similar fails to refresh, hard-reload (Ctrl+Shift+R). For the MJPEG endpoint a `?t=<ts>` query is a reliable cache-bust
- **Sign convention in metric reports.** Showing "fps decreased" as `+71.5%` is misleading. Standardize on "+ = better, - = worse"

---

## 8. Related documents

- [`HANDOFF.md`](./HANDOFF.md) — original Claude.ai design doc translated to English (the input to this implementation)
- [`README.md`](./README.md) — git clone → running, step by step
- [`docs/01_pipeline_architecture.svg`](./docs/01_pipeline_architecture.svg) and [`docs/02_camera_abstraction_layer.svg`](./docs/02_camera_abstraction_layer.svg) — design diagrams
- [`docs/LLaVA設計図.pptx`](./docs/LLaVA設計図.pptx) — original Chrome-side screen layout sketch
- 日本語版: [`HANDOFFJ.md`](./HANDOFFJ.md) / [`READMEJ.md`](./READMEJ.md) / [`TECHNICALJ.md`](./TECHNICALJ.md)
