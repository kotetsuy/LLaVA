"""Read the latest frame from SHM in a separate process. Verifies Steps 1-2.

    uv run shm-reader-demo                 # prints metadata + connected at 5 Hz
    uv run shm-reader-demo --show          # opens a cv2 window (needs GUI)
    uv run shm-reader-demo --save out.jpg  # save one snapshot and exit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

from src.capture.shm_writer import FrameSHM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--show", action="store_true", help="display frames with cv2.imshow")
    parser.add_argument("--save", type=Path, help="save one snapshot to this path and exit")
    parser.add_argument("--ticks", type=int, default=50, help="iterations when not --show/--save")
    parser.add_argument("--rate-hz", type=float, default=5.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["shm"]["name"]

    print(f"attaching SHM {name!r}...")
    shm = FrameSHM.attach(name)
    print(
        f"attached. shape=({shm.frame_h},{shm.frame_w},{shm.channels}) "
        f"pixel_format={shm.pixel_format}"
    )

    last_seq = -1
    last_connected: bool | None = None
    period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    try:
        if args.save:
            for _ in range(20):
                got = shm.read()
                if got is not None:
                    frame, meta = got
                    cv2.imwrite(str(args.save), frame)
                    print(
                        f"saved {args.save} (seq={meta.seq}, ts_ns={meta.timestamp_ns}, "
                        f"connected={meta.connected})"
                    )
                    return 0
                time.sleep(0.05)
            print("timed out waiting for first frame", file=sys.stderr)
            return 1

        if args.show:
            while True:
                got = shm.read()
                if got is not None:
                    frame, meta = got
                    if meta.connected != last_connected:
                        print(f"connected -> {meta.connected} (seq={meta.seq})")
                        last_connected = meta.connected
                    last_seq = meta.seq
                    cv2.imshow("shm reader demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
            return 0

        for tick in range(args.ticks):
            got = shm.read()
            if got is None:
                print(f"[{tick:>3}] no frame")
            else:
                _, meta = got
                fresh = "*" if meta.seq != last_seq else " "
                edge = ""
                if meta.connected != last_connected:
                    edge = f"  <-- connected={meta.connected}"
                    last_connected = meta.connected
                last_seq = meta.seq
                print(
                    f"[{tick:>3}]{fresh} seq={meta.seq:>6} connected={int(meta.connected)} "
                    f"orig={meta.original_w}x{meta.original_h} "
                    f"pad=({meta.pad_x},{meta.pad_y}) scale={meta.scale:.4f}{edge}"
                )
            if period:
                time.sleep(period)
    finally:
        shm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
