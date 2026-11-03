import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import DetectorConfig  # noqa: E402
from heartbeat.detection import locate_r_peaks  # noqa: E402
from heartbeat.fusion import ChannelDetection, fuse_primary, pulse_delay  # noqa: E402

CFG = DetectorConfig()
FS = CFG.target_fs
EMPTY = np.array([], dtype=int)


def regular_beats(n=60, rr=200, start=100):
    return start + rr * np.arange(n)


def test_r_peak_moves_to_largest_deflection():
    ecg = np.zeros(1000)
    ecg[480] = -3.0  # inverted complex
    assert list(locate_r_peaks(ecg, np.array([500]), before=25, after=12)) == [480]


def test_pulse_delay_is_recovered():
    ecg = regular_beats()
    pulse = ecg + 60
    assert np.allclose(pulse_delay(ecg, pulse, CFG), 60)


def test_gap_in_ecg_is_filled_from_pulse():
    truth = regular_beats()
    ecg_beats = np.delete(truth, np.arange(20, 30))  # lead off for 10 beats
    ecg = ChannelDetection(0, "ecg", ecg_beats, EMPTY, 1.0, "ECG")
    pulse = ChannelDetection(1, "bp", truth + 60, EMPTY, 1.0, "ABP")

    fused = fuse_primary(ecg, pulse, CFG)
    assert len(fused) == len(truth)
    assert np.max(np.abs(fused - truth)) <= 2


def test_noisy_ecg_window_is_replaced():
    truth = regular_beats()
    rng = np.random.default_rng(0)
    noise = np.sort(rng.choice(np.arange(4100, 6100), size=15, replace=False))
    ecg_beats = np.concatenate([truth[truth < 4100], noise, truth[truth >= 6100]])
    ecg = ChannelDetection(0, "ecg", ecg_beats, EMPTY, 1.0, "ECG")
    pulse = ChannelDetection(1, "bp", truth + 60, EMPTY, 1.0, "ABP")

    fused = fuse_primary(ecg, pulse, CFG)
    # everything inside the replaced window comes from the pulse channel now
    window = (fused >= 5 * FS) & (fused < 6 * FS)
    assert np.all(np.isin(fused[window], truth))


def test_clean_ecg_is_left_alone():
    truth = regular_beats()
    ecg = ChannelDetection(0, "ecg", truth, EMPTY, 1.0, "ECG")
    pulse = ChannelDetection(1, "bp", truth + 60, EMPTY, 1.0, "ABP")
    assert np.array_equal(fuse_primary(ecg, pulse, CFG), truth)
