"""Step 4 benchmark: 1 image -> 1 Japanese caption with Nemotron-3 Nano Omni.

HANDOFF section 7 step 4: ``llama.cpp で 1 回の画像推論時間を測定 (2 秒に間に合うか)``.

Each ``llama-mtmd-cli`` invocation reloads the 21 GB GGUF; we report
``inference_ms = prompt_eval + eval`` separately (the production-relevant
number, since Step 5 will use a persistent ``llama-server``). ``load_ms`` is
shown for context but should be ignored for the 2 s budget — it's amortized
once at startup in the real pipeline.

Usage::

    # save a snapshot from the live capture process and benchmark on it
    uv run capture-run &
    uv run shm-reader-demo --save /tmp/snap.jpg
    uv run benchmark-vlm --image /tmp/snap.jpg

    # or any image file you have lying around
    uv run benchmark-vlm --image ./test.jpg --runs 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import yaml

from src.inference.vlm_worker import VlmWorker


def _fmt_ms(v: float) -> str:
    if v != v:  # NaN
        return "    n/a"
    return f"{v:8.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--image", type=Path, required=True, help="image file to caption")
    parser.add_argument("--prompt", default=None, help="override config.vlm.prompt")
    parser.add_argument("--runs", type=int, default=1, help="number of runs (page cache warms after run 1)")
    parser.add_argument(
        "--show-stderr",
        action="store_true",
        help="print full llama.cpp stderr (timing log, debug info)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: --image {args.image} does not exist", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text())
    vc = cfg["vlm"]
    worker = VlmWorker(
        binary=vc["binary"],
        model=vc["model"],
        mmproj=vc["mmproj"],
        prompt=args.prompt or vc["prompt"],
        ngl=vc.get("ngl", 99),
        ctx_size=vc.get("ctx_size", 8192),
        n_predict=vc.get("n_predict", 96),
        temp=vc.get("temp", 0.2),
        extra_args=list(vc.get("extra_args", [])),
    )

    print(f"image:  {args.image}")
    print(f"prompt: {worker.prompt}")
    print(f"model:  {Path(worker.model).name}")
    print(f"mmproj: {Path(worker.mmproj).name}")
    print(f"ngl={worker.ngl} ctx={worker.ctx_size} n_predict={worker.n_predict} temp={worker.temp}")
    print()

    timings = []
    for i in range(args.runs):
        print(f"=== run {i+1}/{args.runs} ===")
        result = worker.predict_image(args.image)
        if result.returncode != 0:
            print(f"  llama-mtmd-cli failed (rc={result.returncode})", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return result.returncode

        if args.show_stderr:
            print("--- stderr ---")
            print(result.stderr)
            print("--- /stderr ---")

        t = result.timing
        print(f"  caption: {result.caption!r}")
        print(
            f"  load           {_fmt_ms(t.load_ms)}                    "
            f"(amortized in production)"
        )
        print(
            f"  prompt eval    {_fmt_ms(t.prompt_eval_ms)}  "
            f"({t.n_prompt_tokens:>5} tokens, {t.prompt_eval_tps:6.1f} t/s)"
        )
        print(
            f"  eval           {_fmt_ms(t.eval_ms)}  "
            f"({t.n_eval_tokens:>5} tokens, {t.eval_tps:6.1f} t/s)"
        )
        print(f"  inference (no-load) {_fmt_ms(t.inference_ms)}    <-- this is the 2s budget number")
        print(f"  total (mtmd-cli internal) {_fmt_ms(t.total_ms)}")
        print(f"  wallclock       {_fmt_ms(t.wall_ms)}")
        print()
        timings.append(t)

    if args.runs > 1:
        infs = [t.inference_ms for t in timings if t.inference_ms == t.inference_ms]
        if len(infs) >= 2:
            print(
                f"inference_ms across runs: min={min(infs):.1f}  median={statistics.median(infs):.1f}  "
                f"max={max(infs):.1f}"
            )

    # Go/No-Go for Step 4
    last_inf = timings[-1].inference_ms
    if last_inf == last_inf:
        if last_inf <= 2000:
            print(
                f"\nRESULT: inference={last_inf:.0f}ms <= 2000ms  "
                f"-> Step 4 PASS (Go on Step 5: YOLO + VLM concurrent)"
            )
        else:
            print(
                f"\nRESULT: inference={last_inf:.0f}ms > 2000ms  "
                f"-> reduce n_predict, try smaller quant, or shorter prompt"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
