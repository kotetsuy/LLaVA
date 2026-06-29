"""Pure-logic tests for camera priority ranking and selection.

These exercise ``priority_rank`` / ``select_device_ranked`` without touching
pyudev or cv2: we build ``CameraDevice`` dataclasses directly and disable the
capture probe with ``capture_check=False``.
"""

from __future__ import annotations

from src.capture.device_manager import (
    CameraDevice,
    priority_rank,
    select_device,
    select_device_ranked,
)

JIELI = CameraDevice(
    by_id="usb-Jieli_Technology_USB_PHY_2.0-video-index0",
    dev_path="/dev/video0",
    vid="1124",
    pid="2925",
)
LOGI = CameraDevice(by_id=None, dev_path="/dev/video2", vid="046d", pid="0892")
UNKNOWN = CameraDevice(by_id=None, dev_path="/dev/video4", vid="dead", pid="beef")

PREFERRED = [
    {"name": "jieli-webcam", "by_id": "usb-Jieli_Technology_USB_PHY_2.0-video-index0"},
    {"name": "jieli-vidpid", "vid_pid": "1124:2925"},
    {"name": "logicool-c920", "vid_pid": "046d:0892"},
]


def _select(devices, fallback="any", exclude=()):  # capture probe disabled
    return select_device_ranked(
        devices, PREFERRED, fallback=fallback, capture_check=False, exclude_dev_paths=exclude
    )


def test_priority_rank_matches_first_entry():
    assert priority_rank(JIELI, PREFERRED) == 0  # by_id wins over the vid_pid alias
    assert priority_rank(LOGI, PREFERRED) == 2
    assert priority_rank(UNKNOWN, PREFERRED) is None


def test_highest_priority_wins_when_both_present():
    dev, rank = _select([LOGI, JIELI])
    assert dev is JIELI and rank == 0


def test_lower_priority_used_when_alone():
    dev, rank = _select([LOGI])
    assert dev is LOGI and rank == 2


def test_fallback_any_accepts_unlisted():
    dev, rank = _select([UNKNOWN])
    assert dev is UNKNOWN and rank is None


def test_fallback_none_rejects_unlisted():
    assert _select([UNKNOWN], fallback="none") is None


def test_exclude_active_device_drops_to_next_candidate():
    # Jieli is active and excluded -> only the unlisted camera remains.
    dev, rank = _select([JIELI, UNKNOWN], exclude=("/dev/video0",))
    assert dev is UNKNOWN and rank is None


def test_select_device_wrapper_matches_ranked():
    assert select_device([LOGI, JIELI], PREFERRED, capture_check=False) is JIELI
    assert select_device([UNKNOWN], PREFERRED, fallback="none", capture_check=False) is None
