import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import DetectorConfig  # noqa: E402
from heartbeat.detection import locate_r_peaks  # noqa: E402
from heartbeat.fusion import ChannelDetection, correct_delay, fuse_primary  # noqa: E402

CFG = DetectorConfig()
FS = CFG.target_fs
EMPTY = np.array([], dtype=int)


def regular_beats(n=60, rr=200, start=100):
    return start + rr * np.arange(n)


def test_r_peak_moves_to_largest_deflection():
    ecg = np.zeros(1000)
    ecg[480] = -3.0  # inverted complex
    assert list(locate_r_peaks(ecg, np.array([500]), before=25, after=12)) == [480]


def test_delay_correction():
    ecg = regular_beats()
    assert np.array_equal(correct_delay(ecg, ecg + 60, 19, 125), ecg)


def test_gap_in_ecg_is_filled_from_pulse():
    truth = regular_beats()
    ecg_beats = np.delete(truth, np.arange(20, 30))  # lead off for 10 beats
    ecg = ChannelDetection(0, "ecg", ecg_beats, EMPTY, 1.0, "ECG")
    pulse = ChannelDetection(1, "bp", truth + 60, EMPTY, 1.0, "ABP")

    fused = fuse_primary(ecg, pulse, CFG)
    assert len(fused) == len(truth)
    assert np.max(np.abs(fused - truth)) <= 2


def test_clean_ecg_is_left_alone():
    truth = regular_beats()
    ecg = ChannelDetection(0, "ecg", truth, EMPTY, 1.0, "ECG")
    pulse = ChannelDetection(1, "bp", truth + 60, EMPTY, 1.0, "ABP")
    assert np.array_equal(fuse_primary(ecg, pulse, CFG), truth)
