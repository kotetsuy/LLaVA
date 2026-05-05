"""Configure a ``cv2.VideoCapture`` to a desired format.

Step 1 keeps this small: we just call the V4L2 ``set()`` ladder and read back
what the driver actually accepted. Step 2 will add ``v4l2-ctl --list-formats-ext``
parsing for proper "closest match" negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class NegotiatedFormat:
    fourcc: str
    width: int
    height: int
    fps: float


def _fourcc_int(fourcc: str) -> int:
    if len(fourcc) != 4:
        raise ValueError(f"FourCC must be 4 chars, got {fourcc!r}")
    return cv2.VideoWriter_fourcc(*fourcc)


def _read_back_fourcc(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def configure(
    cap: cv2.VideoCapture,
    fourcc_priority: list[str],
    width: int,
    height: int,
    fps: int,
) -> NegotiatedFormat:
    last_fourcc = ""
    for fourcc in fourcc_priority:
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc_int(fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_fourcc = _read_back_fourcc(cap)
        last_fourcc = actual_fourcc
        if actual_fourcc == fourcc:
            break
    return NegotiatedFormat(
        fourcc=last_fourcc,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
    )
