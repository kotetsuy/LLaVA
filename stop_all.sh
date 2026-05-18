#!/usr/bin/env bash
# Stop the LLaVA tmux session created by start_all.sh.
# Sends Ctrl-C to each window for graceful shutdown (capture releases the
# camera, serve closes the MJPEG stream and WebSocket channels, llama-server
# unloads), waits, then kill-sessions whatever is left.

set -uo pipefail   # not -e: we want to continue on tmux errors

SESSION=llava
GRACEFUL_WAIT=5    # seconds to wait for processes to clean up

if ! command -v tmux >/dev/null; then
    echo "tmux not installed; nothing to stop"
    exit 0
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is not running"
    exit 0
fi

echo "sending SIGINT to capture / serve / vlm windows..."
for win in capture serve vlm; do
    if tmux list-windows -t "$SESSION" -F "#{window_name}" 2>/dev/null | grep -qx "$win"; then
        tmux send-keys -t "$SESSION:$win" C-c
    fi
done

echo "waiting ${GRACEFUL_WAIT}s for graceful shutdown..."
sleep "$GRACEFUL_WAIT"

echo "killing tmux session '$SESSION'..."
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Best-effort: if any of our processes survived (orphaned because the user
# detached and the session was killed without C-c reaching them), clean up.
# We deliberately use -INT first; -KILL only if a stragglers list is left.
pgrep -f "src.server.app|capture.main|llama-server" >/dev/null && {
    echo "post-cleanup: SIGINT to lingering pipeline processes..."
    pkill -INT -f "src.server.app|capture.main|llama-server" || true
    sleep 2
    pgrep -f "src.server.app|capture.main|llama-server" >/dev/null && {
        echo "post-cleanup: SIGKILL holdouts..."
        pkill -KILL -f "src.server.app|capture.main|llama-server" || true
    }
} || true

echo "done."
