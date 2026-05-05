"""Export YOLO11m to ONNX (HANDOFF 5 退避プラン).

Usage::

    uv run export-yolo-onnx                         # exports to <model>.onnx
    uv run export-yolo-onnx --output yolo11m.onnx
    uv run export-yolo-onnx --verify                # also smoke-test with onnxruntime

MIGraphX provider needs an AMD-built onnxruntime; ``pip install onnxruntime``
gives you only ``CPUExecutionProvider``. The verify step picks MIGraphX if
available, otherwise CPU — letting us see numbers either way.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _import_yolo():
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO
    except ImportError as e:
        print(f"ERROR: {e}\n  uv pip install -e .[yolo]", file=sys.stderr)
        sys.exit(2)


def _verify_with_ort(onnx_path: str, imgsz: int) -> int:
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: {e}\n  uv pip install -e .[onnx]", file=sys.stderr)
        return 2

    print(f"\nonnxruntime providers available: {ort.get_available_providers()}")
    providers: list[str] = []
    for cand in ("MIGraphXExecutionProvider", "ROCMExecutionProvider"):
        if cand in ort.get_available_providers():
            providers.append(cand)
            break
    providers.append("CPUExecutionProvider")
    print(f"using providers: {providers}")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    in_meta = sess.get_inputs()[0]
    print(f"input: {in_meta.name} shape={in_meta.shape} dtype={in_meta.type}")

    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(5):
        sess.run(None, {in_meta.name: dummy})

    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        outputs = sess.run(None, {in_meta.name: dummy})
    dt = time.perf_counter() - t0
    print(f"{n} runs in {dt*1000:.1f}ms -> avg {dt*1000/n:.2f}ms/frame, {n/dt:.1f} fps")
    print(f"output shapes: {[o.shape for o in outputs]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model", default=None, help="overrides config.yolo.model")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch/spatial axes")
    parser.add_argument("--half", action="store_true", help="export fp16 weights")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify", action="store_true", help="smoke-test with onnxruntime")
    args = parser.parse_args()

    YOLO = _import_yolo()

    cfg = yaml.safe_load(args.config.read_text())
    yc = cfg.get("yolo", {})
    model_path = args.model or yc.get("model", "yolo11m.pt")
    imgsz = args.imgsz or yc.get("imgsz", 640)

    print(f"loading {model_path}...")
    m = YOLO(model_path)
    print(
        f"exporting -> ONNX (imgsz={imgsz} opset={args.opset} "
        f"dynamic={args.dynamic} half={args.half} simplify={args.simplify})"
    )
    out = m.export(
        format="onnx",
        imgsz=imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
    )
    onnx_path = Path(out)
    print(f"wrote {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if args.output and onnx_path != args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.rename(args.output)
        onnx_path = args.output
        print(f"moved -> {onnx_path}")

    if args.verify:
        rc = _verify_with_ort(str(onnx_path), imgsz)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
