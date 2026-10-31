import numpy as np


def match_beats(detected: np.ndarray, reference: np.ndarray, tolerance: int):
    """Greedy one-to-one matching of detected beats to reference beats.

    A detection counts as a true positive if the closest reference beat is
    within `tolerance` samples and has not been matched yet.
    Returns (TP, FP, FN).
    """
    detected = np.asarray(detected)
    reference = np.asarray(reference)
    if len(reference) == 0:
        return 0, len(detected), 0

    used = np.zeros(len(reference), dtype=bool)
    tp = 0
    for d in detected:
        dist = np.abs(reference - d)
        k = np.argmin(dist)
        if dist[k] <= tolerance and not used[k]:
            used[k] = True
            tp += 1
    return tp, len(detected) - tp, len(reference) - tp


def scores(tp: int, fp: int, fn: int) -> dict:
    se = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
    return {"Se": 100 * se, "+P": 100 * ppv, "F1": 100 * f1}
