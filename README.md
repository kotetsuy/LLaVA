# LLaVA on ROCm — USB camera × YOLO11m × Nemotron Nano Omni × Chrome WebRTC

A demo that runs on the NucBox EVO X2 (Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.2.1): live USB-camera video is streamed to Chrome over low-latency WebRTC, with real-time YOLO11m bbox overlay (30 fps) and Japanese captions from Nemotron Nano Omni (0.5 fps).

For design details see [`HANDOFF.md`](./HANDOFF.md) and [`TECHNICAL.md`](./TECHNICAL.md).
日本語版は [`READMEJ.md`](./READMEJ.md) / [`HANDOFFJ.md`](./HANDOFFJ.md) / [`TECHNICALJ.md`](./TECHNICALJ.md).

---

## Requirements

| Item | Expected value |
|------|------|
| Machine | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151, 48 GB unified) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.1 (symlinked at `/opt/rocm`) |
| Python | 3.12 |
| Package manager | `uv` (typically at `~/.local/bin/uv`) |
| USB camera | Any UVC-compliant device |
| Chrome | Any recent build (same machine or another LAN host) |

The following must already be installed:

- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `tmux` (`sudo apt install tmux`)
- ROCm 7.2.1 (`sudo apt install rocm` or AMD's official path)
- llama.cpp ROCm/HIP build (`~/llama.cpp/build/bin/llama-server` and `llama-mtmd-cli` already built)

---

## Setup steps

### 1. Clone the repository

```bash
git clone <this-repo-url> ~/LLaVA
cd ~/LLaVA
```

### 2. Python venv and base dependencies

```bash
uv venv
uv sync
```

This installs `numpy / opencv-python / pyyaml / pyudev` and gets you the state where Step 1 (USB → SHM) and Step 2 (hot-plug-aware CAL) work.

### 3. ROCm PyTorch (required from Step 3 onwards)

PyPI's `torch` is a CUDA build and won't work. Fetch the AMD ROCm wheels directly:

```bash
mkdir -p ~/wheels && cd ~/wheels
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

cd ~/LLaVA
uv pip install ~/wheels/torch-*.whl ~/wheels/torchvision-*.whl \
               ~/wheels/torchaudio-*.whl ~/wheels/triton-*.whl
```

### 4. YOLO + ONNX (fallback plan)

```bash
uv pip install -e .[yolo,onnx]
```

This pulls in `ultralytics` (which auto-downloads YOLO11m on the first `predict`) and `onnx / onnxruntime` (CPU build).

### 5. WebRTC (aiortc + FastAPI)

```bash
uv pip install -e .[webrtc]
```

### 6. ROCm environment variables

Add these to `~/.bashrc` so new shells pick them up automatically:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

(`start_all.sh` re-exports these inside each tmux pane, so you're covered even if you forget to set them in your shell.)

### 7. Fetch the Nemotron Nano Omni GGUF

```bash
mkdir -p ~/nemotron-3
cd ~/nemotron-3

# unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF (Q4_K_XL)
huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF

huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  mmproj-F16.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
```

Around 24.5 GB total. Confirm that `vlm.model` / `vlm.mmproj` in `config.yaml` match these paths.

### 8. (Optional) confirm the GPU is visible

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Radeon Graphics
```

---

## Start / stop

### One-shot launch (recommended)

```bash
cd ~/LLaVA
./start_all.sh
```

This single command:

1. Creates the tmux session `llava`
2. window 0: `uv run capture-run` (USB → SHM, Steps 1+2)
3. window 1: `uv run serve` (FastAPI + aiortc + YOLO bbox + VLM caption WS, Steps 6+7)
4. window 2: `llama-server --reasoning off` (resident Nemotron, Step 7b)
5. Polls `http://localhost:8080/` for up to 30 seconds
6. Opens Chrome automatically

Options:

```bash
./start_all.sh --no-browser     # over SSH or when you don't want auto-open
./start_all.sh --help
```

Attach to the session:

```bash
tmux attach -t llava            # watch logs
# Ctrl-b 0 / 1 / 2 to switch windows
# Ctrl-b d to detach but keep the session running
```

### Stop

```bash
./stop_all.sh
```

Sends `Ctrl-C` to each window, waits 5 seconds, then `tmux kill-session`. If anything survived, it falls through SIGINT → SIGKILL cleanup.

---

## What you should see in the browser

After `./start_all.sh`, open `http://localhost:8080/` in Chrome (or `http://<NucBox-IP>:8080/` from another LAN host):

- A `<video>` showing the live camera (1280×720, ≤150 ms latency)
- A semi-transparent `<canvas>` overlay with color-coded YOLO bboxes (30 fps; labels like `person 92%`)
- A semi-transparent caption box below with the Nemotron Japanese caption (~50 chars, refreshed every 2 s)
- A status row showing the WebRTC connection state, bbox WS state, and the latest VLM `inference_ms` and `t/s`

Captions stay at `(no caption yet)` for the first ~10 s while llama-server loads the GGUF, then updates kick in.

---

## Step-by-step (for debugging)

If you want to bring components up one at a time without `start_all.sh`:

```bash
# Steps 1+2: capture
uv run capture-run                          # in another terminal
uv run shm-reader-demo --ticks 10           # confirm SHM read path
uv run shm-reader-demo --save /tmp/snap.jpg # save a single frame
uv run list-cameras                         # list /dev/v4l devices

# Step 3: YOLO standalone
uv run benchmark-yolo --source synthetic    # synthetic 1280x720 noise
uv run benchmark-yolo --source shm          # against the running capture-run
uv run export-yolo-onnx --verify            # ONNX fallback

# Step 4: VLM standalone (mtmd-cli subprocess)
uv run benchmark-vlm --image /tmp/snap.jpg

# Step 5: YOLO + VLM concurrent
uv run benchmark-concurrent --frames 600
uv run benchmark-concurrent --no-vlm        # baseline

# Steps 6+7: server only
uv run serve                                # ≈ T2
~/llama.cpp/build/bin/llama-server -m ... --mmproj ... --reasoning off  # ≈ T3
```

---

## Troubleshooting

### `uv run capture-run` doesn't see a camera

```bash
ls /dev/v4l/by-id              # is the USB camera symlinked?
v4l2-ctl --list-devices        # (sudo apt install v4l-utils)
```

If your camera doesn't match `camera.preferred[].by_id` in `config.yaml`, the `fallback: any` policy still picks something. To force a specific camera, edit `by_id: usb-Vendor_Model*`.

### Chrome shows `connection failed`

A WebRTC ICE failure. Most are transient — `Ctrl+Shift+R` to hard-reload usually fixes it. If you can't connect from another LAN host, open the firewall: `sudo ufw allow 8080`.

### Caption stays empty (`(no caption yet)`)

Either llama-server is still loading (~10 s the first time) or `--reasoning off` was forgotten. Check `serve`'s log via `tmux attach -t llava` → Ctrl-b 1:

```
vlm-runner: caption (1300ms) 'これは...'                  ← OK
vlm-runner: empty caption after strip; raw='<think>...'   ← --reasoning off missing
```

### `start_all.sh` fails with "session already exists"

```bash
./stop_all.sh                  # stop first
./start_all.sh                 # then restart
```

Or `tmux kill-session -t llava` to force-kill.

### Model load is unusually slow (30+ s the first time)

That's the time to pull the 21 GB GGUF off NVMe and into the page cache. Subsequent loads drop to ~10 s.

### Switching YOLO from fp16 back to fp32

Change `yolo.half: true` → `false` in `config.yaml`. fp32 has marginally better accuracy, but Step 5's concurrent benchmark showed fp16 leaves the VLM more headroom, so we default to fp16.

---

## Self-checks during development

```bash
# Byte-compile every Python file
python3 -m compileall -q src scripts && echo OK

# Module import smoke (also resolves dependency closure)
uv run python -c "from src.server.app import app; print('imports OK')"
```

---

## License

See [`LICENSE`](./LICENSE).
