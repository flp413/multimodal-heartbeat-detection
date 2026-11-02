import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import HeartbeatDetector, get_preset  # noqa: E402
from heartbeat.evaluation import match_beats, scores  # noqa: E402

FS = 250


def synthetic_record(seconds=60, rr=0.8, delay=0.25, seed=0):
    """Narrow spikes for the ECG and smooth delayed pulses for the pressure."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    beats = np.arange(0.5, seconds - 1, rr) + rng.normal(0, 0.02, int((seconds - 1.5) / rr) + 1)

    ecg = np.zeros_like(t)
    bp = np.full_like(t, 80.0)
    for b in beats:
        ecg += np.exp(-((t - b) / 0.012) ** 2)
        bp += 40 * np.exp(-((t - b - delay) / 0.08) ** 2)
    ecg += rng.normal(0, 0.02, len(t))
    return np.column_stack([ecg, bp]), np.round(beats * FS).astype(int)


def test_finds_beats_in_synthetic_record():
    signals, truth = synthetic_record()
    beats = HeartbeatDetector(get_preset("improved")).detect(signals, FS, ["ECG", "ABP"])
    s = scores(*match_beats(beats, truth, int(0.05 * FS)))
    assert s["Se"] > 98 and s["+P"] > 98
