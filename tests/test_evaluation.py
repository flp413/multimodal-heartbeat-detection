import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat.detection import enforce_min_rr  # noqa: E402
from heartbeat.evaluation import match_beats, scores  # noqa: E402


def test_perfect_match():
    ref = np.array([100, 300, 500])
    assert match_beats(ref, ref, tolerance=5) == (3, 0, 0)


def test_missed_and_extra_beats():
    ref = np.array([100, 300, 500])
    det = np.array([102, 400, 499])
    assert match_beats(det, ref, tolerance=5) == (2, 1, 1)


def test_two_detections_on_one_beat():
    ref = np.array([100])
    det = np.array([99, 101])
    assert match_beats(det, ref, tolerance=5) == (1, 1, 0)


def test_scores():
    s = scores(tp=90, fp=10, fn=10)
    assert round(s["Se"], 2) == 90.0
    assert round(s["+P"], 2) == 90.0
    assert round(s["F1"], 2) == 90.0


def test_min_rr_drops_close_beats():
    beats = np.array([0, 10, 100, 120, 300])
    assert list(enforce_min_rr(beats, 50)) == [0, 100, 300]
