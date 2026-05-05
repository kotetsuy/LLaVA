"""Thin wrapper around Ultralytics YOLO11m.

Reads a 1280x720 BGR frame (the SHM "normalized" frame), runs detection at
``imgsz`` (640 by default — Ultralytics handles the letterbox to 640x640 and
maps the bbox coordinates back to the input frame), and returns
``Detection``s in the input frame's coordinate system.

Imports of ``torch`` / ``ultralytics`` are deferred until ``__init__`` so
modules that only need the ``Detection`` dataclass (e.g. the WebRTC server)
can import this module without those heavyweight deps installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]  # input-frame coords (typically 1280x720)
    cls: int
    conf: float
    label: str


class YoloWorker:
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # noqa: PLC0415  (deferred for optional dep)

        self._model: "YOLO" = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self._names = self._model.names
        log.info(
            "YoloWorker ready: model=%s device=%s imgsz=%d half=%s classes=%d",
            model_path, device, imgsz, half, len(self._names),
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        out: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs, strict=True):
            out.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    cls=int(c),
                    conf=float(p),
                    label=self._names.get(int(c), str(int(c))),
                )
            )
        return out

    def predict_with_speed(self, frame: np.ndarray) -> tuple[list[Detection], dict[str, float]]:
        """Same as ``predict`` but also returns ultralytics' per-stage timings (ms)."""
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            verbose=False,
        )
        speed = dict(results[0].speed) if results else {}
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return [], speed
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        out = [
            Detection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                cls=int(c),
                conf=float(p),
                label=self._names.get(int(c), str(int(c))),
            )
            for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs, strict=True)
        ]
        return out, speed
