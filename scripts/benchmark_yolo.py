"""Benchmark YOLO11m on synthetic frames or live SHM frames.

Step 3 Go/No-Go: does YOLO11m hit ~30 fps at imgsz=640 on this box?

Examples::

    # synthetic 1280x720 noise frames, fp32 on the GPU
    uv run benchmark-yolo --frames 200

    # live frames from the running capture process
    uv run capture-run &
    uv run benchmark-yolo --source shm --frames 300

    # fp16 sweep
    uv run benchmark-yolo --half

ROCm note: if device='cuda' fails, make sure HSA_OVERRIDE_GFX_VERSION=11.5.1
is exported and HIP_VISIBLE_DEVICES=0 (or unset). See CLAUDE.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _import_torch_and_yolo():
    try:
        import torch  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as e:
        print(f"\nERROR: missing dep ({e}).", file=sys.stderr)
        print("See README.md for the ROCm 7.2.1 wheel URLs, then:", file=sys.stderr)
        print("  uv pip install ~/wheels/torch-*.whl ~/wheels/torchvision-*.whl \\", file=sys.stderr)
        print("                 ~/wheels/torchaudio-*.whl ~/wheels/triton-*.whl", file=sys.stderr)
        print("  uv pip install -e .[yolo]\n", file=sys.stderr)
        sys.exit(2)
    return torch, YOLO


def _synthetic_frames(width: int, height: int, n: int, seed: int = 0) -> list[np.ndarray]:
    """Random noise + a few drawn rectangles. Real-world latency depends on the
    detection count (post-processing/NMS), so a noisy synthetic with ~0 dets
    can *under*-state latency. Prefer ``--source shm`` for true numbers."""
    import cv2  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        # paint a couple of boxes so the model sees structure
        cv2.rectangle(f, (50 + i % 20, 100), (250 + i % 20, 400), (200, 50, 50), -1)
        cv2.circle(f, (700, 300 + i % 40), 80, (50, 200, 80), -1)
        frames.append(f)
    return frames


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = int(len(sorted_vals) * p)
    k = max(0, min(len(sorted_vals) - 1, k))
    return sorted_vals[k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model", default=None, help="overrides config.yolo.model")
    parser.add_argument("--device", default=None, help="overrides config.yolo.device")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--source", choices=["synthetic", "shm"], default="synthetic")
    parser.add_argument("--half", action="store_true", help="fp16 inference")
    parser.add_argument(
        "--unique-frames",
        type=int,
        default=64,
        help="number of distinct synthetic frames cycled through (CPU prepro reuse)",
    )
    args = parser.parse_args()

    torch, YOLO = _import_torch_and_yolo()

    cfg = yaml.safe_load(args.config.read_text())
    yc = cfg.get("yolo", {})
    model_path = args.model or yc.get("model", "yolo11m.pt")
    device = args.device or yc.get("device", "cuda")
    imgsz = args.imgsz or yc.get("imgsz", 640)
    half = args.half or yc.get("half", False)
    target = cfg["camera"]["output"]["target"]

    print(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0: {torch.cuda.get_device_name(0)}")
    print(f"loading model={model_path} device={device} imgsz={imgsz} half={half}")
    model = YOLO(model_path)

    # source
    if args.source == "synthetic":
        pool = _synthetic_frames(target[0], target[1], args.unique_frames)
        get_frame = lambda i: pool[i % len(pool)]  # noqa: E731
        cleanup = lambda: None  # noqa: E731
    else:
        from src.capture.shm_writer import FrameSHM  # noqa: PLC0415

        print(f"attaching SHM {cfg['shm']['name']!r}...")
        shm = FrameSHM.attach(cfg["shm"]["name"])

        def get_frame(_i: int) -> np.ndarray | None:
            got = shm.read()
            return got[0] if got is not None else None

        cleanup = shm.close

    try:
        # warmup
        print(f"warmup {args.warmup} frames...")
        produced = 0
        i = 0
        while produced < args.warmup:
            f = get_frame(i)
            i += 1
            if f is None:
                time.sleep(0.005)
                continue
            model.predict(f, imgsz=imgsz, device=device, half=half, verbose=False)
            produced += 1

        # benchmark
        print(f"benchmarking {args.frames} frames...")
        latencies_ms: list[float] = []
        speed_pre: list[float] = []
        speed_inf: list[float] = []
        speed_post: list[float] = []
        last_n_dets = 0

        produced = 0
        i = 0
        t_total = time.perf_counter()
        while produced < args.frames:
            f = get_frame(i)
            i += 1
            if f is None:
                time.sleep(0.005)
                continue
            t = time.perf_counter()
            results = model.predict(f, imgsz=imgsz, device=device, half=half, verbose=False)
            latencies_ms.append((time.perf_counter() - t) * 1000)
            r = results[0]
            sp = r.speed
            speed_pre.append(sp.get("preprocess", 0.0))
            speed_inf.append(sp.get("inference", 0.0))
            speed_post.append(sp.get("postprocess", 0.0))
            last_n_dets = 0 if r.boxes is None else len(r.boxes)
            produced += 1
        elapsed = time.perf_counter() - t_total

        s = sorted(latencies_ms)
        n = len(latencies_ms)
        fps = n / elapsed if elapsed > 0 else float("nan")
        print()
        print(f"frames={n}  elapsed={elapsed:.2f}s  ->  {fps:.1f} fps")
        print(
            f"end-to-end latency:  p50={_percentile(s, 0.50):.2f}ms  "
            f"p95={_percentile(s, 0.95):.2f}ms  p99={_percentile(s, 0.99):.2f}ms"
        )
        print(
            f"ultralytics speed:   pre={statistics.mean(speed_pre):.2f}ms  "
            f"inf={statistics.mean(speed_inf):.2f}ms  "
            f"post={statistics.mean(speed_post):.2f}ms"
        )
        print(f"last frame detections: {last_n_dets}")
        if fps >= 30:
            print("RESULT: >=30 fps  -> Step 3 PASS (Go on Step 5 wiring)")
        else:
            print(f"RESULT: <30 fps  -> consider --half, smaller imgsz, or YOLO11s/n")
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
