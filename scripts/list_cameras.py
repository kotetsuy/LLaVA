"""List USB cameras seen via udev (subsystem=video4linux).

    uv run list-cameras
"""

from __future__ import annotations

import sys

from src.capture.device_manager import enumerate_devices, is_capture_capable


def main() -> int:
    devs = enumerate_devices()
    if not devs:
        print("(no v4l devices found)")
        return 1
    print(f"{'CAPTURE':<8} {'DEV':<14} {'VID:PID':<11} BY-ID")
    print("-" * 90)
    for d in devs:
        cap_ok = "yes" if is_capture_capable(d.dev_path) else "no"
        print(f"{cap_ok:<8} {d.dev_path:<14} {d.vid_pid or '-':<11} {d.by_id or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
