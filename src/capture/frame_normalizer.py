from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray
    pad_x: int
    pad_y: int
    scale: float
    original_w: int
    original_h: int


def letterbox(
    src: np.ndarray,
    target_w: int = 1280,
    target_h: int = 720,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    h, w = src.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    if (nw, nh) != (w, h):
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = src
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    pad_x = (target_w - nw) // 2
    pad_y = (target_h - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return LetterboxResult(
        image=canvas,
        pad_x=pad_x,
        pad_y=pad_y,
        scale=scale,
        original_w=w,
        original_h=h,
    )
