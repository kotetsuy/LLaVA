#!/usr/bin/env bash
# Start the full LLaVA pipeline in a tmux session named "llava".
# 3 windows (visible via `tmux attach -t llava`):
#   capture  -- USB camera -> SHM (Step 1+2)
#   serve    -- FastAPI + aiortc + YOLO bbox WS + VLM caption WS
#   vlm      -- llama-server (Nemotron-3 Nano Omni multimodal)
#
# Order doesn't matter — yolo-runner and vlm-runner inside `serve` retry
# until SHM and llama-server are reachable. We add a short stagger
# anyway so logs read top-to-bottom on first attach.

set -euo pipefail

SESSION=llava
PROJECT_DIR="$HOME/LLaVA"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
VLM_MODEL="$HOME/nemotron-3/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf"
VLM_MMPROJ="$HOME/nemotron-3/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/mmproj-F16.gguf"
URL="http://localhost:8080/"
SERVER_TIMEOUT=30  # seconds to wait for the server before opening the browser

# CLI options
OPEN_BROWSER=1
for arg in "$@"; do
    case "$arg" in
        --no-browser) OPEN_BROWSER=0 ;;
        -h|--help)
            echo "Usage: $0 [--no-browser]"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ROCm env (mirrors CLAUDE.md whisperx setup; safe to set per-pane).
ENV_PREFIX='export HSA_OVERRIDE_GFX_VERSION=11.5.1 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0; '

if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux is not installed. Install with: sudo apt install tmux" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: project dir not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-server not found at $LLAMA_BIN" >&2
    exit 1
fi
if [[ ! -f "$VLM_MODEL" ]]; then
    echo "ERROR: VLM model not found at $VLM_MODEL" >&2
    exit 1
fi
if [[ ! -f "$VLM_MMPROJ" ]]; then
    echo "ERROR: mmproj not found at $VLM_MMPROJ" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' is already running."
    echo "       Run ./stop_all.sh first, or attach: tmux attach -t $SESSION"
    exit 1
fi

# capture
tmux new-session -d -s "$SESSION" -n capture -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:capture" "${ENV_PREFIX}uv run capture-run" C-m

# Give capture ~1s head-start so SHM is ready before yolo/vlm start probing.
sleep 1

# serve (FastAPI + aiortc + YoloRunner + VlmRunner)
tmux new-window -t "$SESSION:" -n serve -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:serve" "${ENV_PREFIX}uv run serve" C-m

# llama-server (multimodal Nemotron). --reasoning off is required:
# without it the *-Reasoning model spends n_predict on thinking tokens.
tmux new-window -t "$SESSION:" -n vlm -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:vlm" "${ENV_PREFIX}${LLAMA_BIN} \
  -m '$VLM_MODEL' \
  --mmproj '$VLM_MMPROJ' \
  -c 8192 -ngl 99 --port 8081 --host 127.0.0.1 --reasoning off" C-m

cat <<EOF
Started tmux session '$SESSION' with 3 windows: capture, serve, vlm.

  Attach :  tmux attach -t $SESSION
  Switch :  Ctrl-b 0 (capture), Ctrl-b 1 (serve), Ctrl-b 2 (vlm)
  Detach :  Ctrl-b d
  Stop   :  ./stop_all.sh

EOF

# Open Chrome once the server is reachable. Caption may still take ~10s
# more after this point — the page just renders "(no caption yet)" until
# llama-server finishes loading the GGUF.
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
    echo -n "waiting for $URL "
    READY=0
    for ((i = 0; i < SERVER_TIMEOUT; i++)); do
        if curl -sf -o /dev/null --max-time 1 "$URL"; then
            READY=1
            break
        fi
        echo -n "."
        sleep 1
    done
    echo
    if [[ "$READY" -eq 0 ]]; then
        echo "WARNING: server didn't respond within ${SERVER_TIMEOUT}s; check 'tmux attach -t $SESSION'."
        echo "Open manually: $URL"
        exit 0
    fi

    if command -v google-chrome >/dev/null 2>&1; then
        google-chrome "$URL" >/dev/null 2>&1 &
        echo "opened in google-chrome."
    elif command -v chromium >/dev/null 2>&1; then
        chromium "$URL" >/dev/null 2>&1 &
        echo "opened in chromium."
    elif command -v chromium-browser >/dev/null 2>&1; then
        chromium-browser "$URL" >/dev/null 2>&1 &
        echo "opened in chromium-browser."
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 &
        echo "opened via xdg-open (default browser)."
    else
        echo "no browser command found; open $URL manually."
    fi
else
    echo "Browser auto-open skipped (--no-browser). Open: $URL"
fi
