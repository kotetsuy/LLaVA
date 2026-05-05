"""Step 5 Go/No-Go: YOLO + VLM concurrent on the same GPU.

Runs YOLO at full pace on the main thread (driven by SHM frames from a
separately-running ``capture-run``) while a background thread fires VLM
inferences at a configurable cadence (default 2s). Reports YOLO fps,
VLM inference_ms, and eval-tok/s — all compared against the Step 3 / 4
single-model baselines.

Usage::

    # in another terminal:
    uv run capture-run

    # then:
    uv run benchmark-concurrent --frames 600

    # YOLO-only baseline check (no VLM thread):
    uv run benchmark-concurrent --no-vlm --frames 600

Caveat: this build uses ``llama-mtmd-cli`` subprocess-per-call for VLM, so
each VLM call reloads the model (page cache makes the second-and-onward
loads cheap). The contention pattern is therefore *bursty* — VLM occupies
the GPU for ~1.5s out of every cadence interval. Production (Step 6+) will
keep VLM resident via ``llama-server`` and contention will be steadier.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import yaml

from src.capture.shm_writer import FrameSHM

# Baselines on this box.
# YOLO_GPU_FPS is the GPU-bound max from Step 3 benchmark-yolo (re-runs on
# duplicate SHM frames, so it measures raw GPU throughput, not the pipeline).
# YOLO_PIPELINE_FPS is the camera-rate-limited number you get when the
# concurrent benchmark runs with ``--no-vlm`` (dedup'd via meta.seq, so
# capped at the capture FPS ~30). The right baseline for Step 5 verdict is
# YOLO_PIPELINE_FPS — that's what production sees.
YOLO_GPU_FPS = 97.8         # Step 3 (no dedup, GPU-bound)
YOLO_PIPELINE_FPS = 30.1    # 2026-05-05 --no-vlm run on this box
VLM_BASELINE_INF_MS = 1262.0
VLM_BASELINE_EVAL_TPS = 161.0

log = logging.getLogger("step5")


def _import_workers():
    try:
        from src.inference.yolo_worker import YoloWorker  # noqa: PLC0415
        from src.inference.vlm_worker import VlmWorker  # noqa: PLC0415
    except ImportError as e:
        print(
            f"ERROR: {e}\n  uv pip install -e .[yolo]  (and torch wheels per README)",
            file=sys.stderr,
        )
        sys.exit(2)
    return YoloWorker, VlmWorker


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p)))
    return sorted_vals[k]


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--frames", type=int, default=600, help="YOLO frames to measure")
    parser.add_argument("--yolo-warmup", type=int, default=20)
    parser.add_argument(
        "--vlm-cadence-sec",
        type=float,
        default=2.0,
        help="seconds between VLM inferences (clamps to actual finish time if longer)",
    )
    parser.add_argument(
        "--no-vlm", action="store_true", help="skip the VLM thread (YOLO-only baseline check)"
    )
    parser.add_argument(
        "--vlm-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run one VLM call before starting measurement (page-caches the GGUF)",
    )
    args = parser.parse_args()

    YoloWorker, VlmWorker = _import_workers()

    cfg = yaml.safe_load(args.config.read_text())
    yc = cfg["yolo"]
    vc = cfg["vlm"]
    shm_name = cfg["shm"]["name"]

    # SHM
    try:
        shm = FrameSHM.attach(shm_name)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: cannot attach SHM {shm_name!r}: {e}", file=sys.stderr)
        print("Start the capture process first: uv run capture-run", file=sys.stderr)
        return 3
    print(f"attached SHM {shm_name!r}: {shm.frame_h}x{shm.frame_w} {shm.pixel_format}")

    # YOLO
    print(f"loading YOLO ({yc['model']}, device={yc['device']}, imgsz={yc['imgsz']})...")
    yolo = YoloWorker(
        model_path=yc["model"],
        device=yc["device"],
        imgsz=yc["imgsz"],
        conf=yc["conf"],
        iou=yc["iou"],
        half=yc.get("half", False),
    )

    # VLM (always built — even with --no-vlm so config issues surface early)
    vlm = VlmWorker(
        binary=vc["binary"],
        model=vc["model"],
        mmproj=vc["mmproj"],
        prompt=vc["prompt"],
        ngl=vc.get("ngl", 99),
        ctx_size=vc.get("ctx_size", 8192),
        n_predict=vc.get("n_predict", 96),
        temp=vc.get("temp", 0.2),
        extra_args=list(vc.get("extra_args", [])),
    )

    tmp_jpg = Path("/tmp/concurrent_vlm.jpg")

    def _snap_to_jpg() -> bool:
        got = shm.read()
        if got is None:
            return False
        frame, _meta = got
        return bool(cv2.imwrite(str(tmp_jpg), frame))

    # YOLO warmup
    print(f"YOLO warmup ({args.yolo_warmup} frames)...")
    warmed = 0
    while warmed < args.yolo_warmup:
        got = shm.read()
        if got is None:
            time.sleep(0.005)
            continue
        yolo.predict(got[0])
        warmed += 1

    # VLM warmup (single call, page-caches the GGUF)
    if not args.no_vlm and args.vlm_warmup:
        print("VLM warmup (1 call, runs alone)...")
        if not _snap_to_jpg():
            print("ERROR: SHM has no frame yet; is capture-run running?", file=sys.stderr)
            return 4
        r = vlm.predict_image(tmp_jpg)
        if r.returncode != 0:
            print(f"VLM warmup failed (rc={r.returncode}):\n{r.stderr}", file=sys.stderr)
            return r.returncode
        print(
            f"  warmup ok: inf={r.timing.inference_ms:.0f}ms  "
            f"eval_tps={r.timing.eval_tps:.1f}  caption={r.caption!r}"
        )

    # VLM background thread
    stop = threading.Event()
    vlm_lock = threading.Lock()
    vlm_timings: list = []
    vlm_errors = 0

    def _vlm_loop() -> None:
        nonlocal vlm_errors
        next_at = time.monotonic()
        while not stop.is_set():
            wait = next_at - time.monotonic()
            if wait > 0:
                if stop.wait(timeout=wait):
                    return
            if not _snap_to_jpg():
                next_at = time.monotonic() + 0.1
                continue
            try:
                result = vlm.predict_image(tmp_jpg, timeout=120.0)
            except Exception as e:  # noqa: BLE001
                log.warning("VLM call raised: %s", e)
                vlm_errors += 1
                next_at = time.monotonic() + args.vlm_cadence_sec
                continue
            if result.returncode != 0:
                log.warning("VLM rc=%d", result.returncode)
                vlm_errors += 1
                next_at = time.monotonic() + args.vlm_cadence_sec
                continue
            with vlm_lock:
                vlm_timings.append(result.timing)
                idx = len(vlm_timings)
            t = result.timing
            print(
                f"  [vlm #{idx}] inf={t.inference_ms:.0f}ms "
                f"(peval={t.prompt_eval_ms:.0f}+eval={t.eval_ms:.0f}) "
                f"eval_tps={t.eval_tps:.1f}"
            )
            next_at = time.monotonic() + args.vlm_cadence_sec

    vlm_thread: threading.Thread | None = None
    if not args.no_vlm:
        vlm_thread = threading.Thread(target=_vlm_loop, name="vlm", daemon=True)
        vlm_thread.start()

    # YOLO measurement loop
    label = "YOLO + VLM" if not args.no_vlm else "YOLO alone (baseline check)"
    print(f"\nrunning {label} on {args.frames} frames...")
    last_seq = -1
    latencies: list[float] = []
    speed_pre: list[float] = []
    speed_inf: list[float] = []
    speed_post: list[float] = []
    t_start = time.perf_counter()
    while len(latencies) < args.frames:
        got = shm.read()
        if got is None:
            time.sleep(0.001)
            continue
        frame, meta = got
        if meta.seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = meta.seq
        t0 = time.perf_counter()
        _, sp = yolo.predict_with_speed(frame)
        latencies.append((time.perf_counter() - t0) * 1000)
        speed_pre.append(sp.get("preprocess", 0.0))
        speed_inf.append(sp.get("inference", 0.0))
        speed_post.append(sp.get("postprocess", 0.0))
    elapsed = time.perf_counter() - t_start

    # Stop VLM thread
    stop.set()
    if vlm_thread is not None:
        vlm_thread.join(timeout=120.0)
        if vlm_thread.is_alive():
            print("warning: VLM thread still running after 120s join", file=sys.stderr)

    # Report
    n = len(latencies)
    fps = n / elapsed if elapsed > 0 else float("nan")
    s = sorted(latencies)
    print()
    print("=" * 72)
    print(f"YOLO ({label}):")
    print(f"  frames={n}  elapsed={elapsed:.2f}s  ->  {fps:.1f} fps")
    print(
        f"  end-to-end latency: p50={_percentile(s, 0.50):.2f}ms  "
        f"p95={_percentile(s, 0.95):.2f}ms  p99={_percentile(s, 0.99):.2f}ms"
    )
    print(
        f"  ultralytics:  pre={statistics.mean(speed_pre):.2f}  "
        f"inf={statistics.mean(speed_inf):.2f}  "
        f"post={statistics.mean(speed_post):.2f} ms"
    )
    print()
    if vlm_timings:
        infs = [t.inference_ms for t in vlm_timings if t.inference_ms == t.inference_ms]
        peval = [t.prompt_eval_ms for t in vlm_timings if t.prompt_eval_ms == t.prompt_eval_ms]
        eval_ = [t.eval_ms for t in vlm_timings if t.eval_ms == t.eval_ms]
        eval_tps = [t.eval_tps for t in vlm_timings if t.eval_tps == t.eval_tps]
        print(f"VLM (under YOLO load, n={len(vlm_timings)}, errors={vlm_errors}):")
        print(
            f"  inference_ms: min={min(infs):.0f}  median={statistics.median(infs):.0f}  "
            f"max={max(infs):.0f}  mean={statistics.mean(infs):.0f}"
        )
        if peval:
            print(
                f"  prompt_eval:  min={min(peval):.0f}  median={statistics.median(peval):.0f}  "
                f"max={max(peval):.0f}"
            )
        if eval_:
            print(
                f"  eval:         min={min(eval_):.0f}  median={statistics.median(eval_):.0f}  "
                f"max={max(eval_):.0f}"
            )
        if eval_tps:
            print(
                f"  eval_tps:     min={min(eval_tps):.1f}  "
                f"median={statistics.median(eval_tps):.1f}  max={max(eval_tps):.1f}"
            )
        print()

    # Comparison vs baselines. Sign convention: + means improvement, - means
    # regression — regardless of whether the metric is "higher is better"
    # (fps, tok/s) or "lower is better" (ms). The trailing label disambiguates.
    print("vs baselines (Δ sign: + = better, - = worse):")
    yolo_pipe_delta = (fps - YOLO_PIPELINE_FPS) / YOLO_PIPELINE_FPS * 100
    yolo_gpu_delta = (fps - YOLO_GPU_FPS) / YOLO_GPU_FPS * 100
    print(
        f"  YOLO fps   pipeline-baseline {YOLO_PIPELINE_FPS:6.1f}  -> {fps:6.1f}  "
        f"({yolo_pipe_delta:+6.1f}%)   <-- the meaningful one"
    )
    print(
        f"  YOLO fps   GPU-bound-max     {YOLO_GPU_FPS:6.1f}  -> {fps:6.1f}  "
        f"({yolo_gpu_delta:+6.1f}%)   (camera caps real fps at ~30)"
    )
    if vlm_timings:
        vlm_med = statistics.median(infs)
        # for ms: lower is better, so delta sign needs flipping for "+ = better"
        vlm_delta = (VLM_BASELINE_INF_MS - vlm_med) / VLM_BASELINE_INF_MS * 100
        print(
            f"  VLM ms     baseline          {VLM_BASELINE_INF_MS:6.0f}  -> {vlm_med:6.0f}  "
            f"({vlm_delta:+6.1f}%)"
        )
        if eval_tps:
            tps_med = statistics.median(eval_tps)
            tps_delta = (tps_med - VLM_BASELINE_EVAL_TPS) / VLM_BASELINE_EVAL_TPS * 100
            print(
                f"  VLM tok/s  baseline          {VLM_BASELINE_EVAL_TPS:6.1f}  -> {tps_med:6.1f}  "
                f"({tps_delta:+6.1f}%)"
            )
    print()

    # Verdict. Strict threshold: pipeline fps within 10% of --no-vlm baseline,
    # AND VLM max within 2s. The 30fps absolute floor is informational only —
    # the camera caps it there to begin with.
    yolo_pipe_drop = (YOLO_PIPELINE_FPS - fps) / YOLO_PIPELINE_FPS * 100
    yolo_pass = yolo_pipe_drop <= 10.0
    vlm_pass = (not vlm_timings) or max(infs) <= 2000
    if yolo_pass and vlm_pass:
        print(
            f"RESULT: Step 5 PASS — YOLO drop {yolo_pipe_drop:+.1f}% (<=10%) "
            "and VLM max <=2s under concurrent load."
        )
        print("        Proceed to Step 6 (aiortc + WebRTC) wiring.")
    else:
        print("RESULT: Step 5 needs tuning.")
        if not yolo_pass:
            p99 = _percentile(s, 0.99)
            print(
                f"        - YOLO dropped {yolo_pipe_drop:.1f}% from --no-vlm baseline "
                f"({YOLO_PIPELINE_FPS:.1f} -> {fps:.1f} fps). p99 latency {p99:.0f}ms."
            )
            print(
                "          Cheap knobs (in order): yolo.half=true (fp16, ~half infer time);"
                " --vlm-cadence-sec 4.0 (halve duty cycle); switch VLM to llama-server"
                " (Step 6 plan; kills subprocess reload). Don't reach for ONNX-CPU"
                " 退避プラン unless these all fail."
            )
        if not vlm_pass:
            print(
                f"        - VLM exceeded 2s budget (max={max(infs):.0f}ms). "
                f"Try n_predict={vc.get('n_predict', 96) // 2} or shorter prompt."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
