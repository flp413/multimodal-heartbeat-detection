import numpy as np

from .config import DetectorConfig
from .detection import detect_beats
from .preprocessing import preprocess

PULSATILE_KEYWORDS = ["pressure", "bp", "cvp", "pa", "pap", "art", "pleth", "abp"]


def rr_regularity(signal: np.ndarray, fs: float, signal_type: str, cfg: DetectorConfig) -> float:
    """Share of consecutive RR ratios that fall inside [rr_ratio_low, rr_ratio_high].

    A channel that gives a regular rhythm with given settings is probably
    of that type and clean enough to use.
    """
    beats, _ = detect_beats(preprocess(signal, fs, signal_type, cfg), cfg)
    if len(beats) < 3:
        return 0.0

    rr = np.diff(beats)
    ratios = rr[1:] / rr[:-1]
    stable = np.sum((ratios >= cfg.rr_ratio_low) & (ratios <= cfg.rr_ratio_high))
    return stable / len(ratios)


def classify_channel(signal: np.ndarray, fs: float, name: str, cfg: DetectorConfig):
    """Decide if a channel should be treated as ECG, BP or skipped.

    Only the first cfg.quality_window seconds are used.
    Returns (signal_type, score), signal_type is '' for unusable channels.
    """
    segment = signal[: int(fs * cfg.quality_window)]
    ecg_score = rr_regularity(segment, fs, "ecg", cfg)
    bp_score = rr_regularity(segment, fs, "bp", cfg)

    ecg_ok = ecg_score >= cfg.min_quality_score
    bp_ok = bp_score >= cfg.min_quality_score

    if ecg_ok and bp_ok:
        # both settings work, fall back to the channel name
        if any(k in name.lower() for k in PULSATILE_KEYWORDS):
            return "bp", bp_score
        return "ecg", ecg_score
    if ecg_ok:
        return "ecg", ecg_score
    if bp_ok:
        return "bp", bp_score
    return "", 0.0
