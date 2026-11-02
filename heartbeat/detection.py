import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import DetectorConfig


def _centered_envelope(signal: np.ndarray, window: int, func) -> np.ndarray:
    half = window // 2
    padded = np.pad(signal, half, mode="edge")
    windows = sliding_window_view(padded, 2 * half)[: len(signal)]
    return func(windows, axis=1)


def dynamic_threshold(rs: np.ndarray, cfg: DetectorConfig):
    """Threshold halfway between the smoothed local min and max of the range signal.

    Also returns the envelope width, which is small in flat or noisy parts.
    """
    window = int(cfg.envelope_window * cfg.target_fs)
    local_min = _centered_envelope(rs, window, np.min)
    local_max = _centered_envelope(rs, window, np.max)

    kernel = np.ones(cfg.smoothing_samples) / cfg.smoothing_samples
    smooth_min = np.convolve(local_min, kernel, mode="same")
    smooth_max = np.convolve(local_max, kernel, mode="same")

    return 0.5 * (smooth_max + smooth_min), smooth_max - smooth_min


def refine_peaks(rs: np.ndarray, candidates: np.ndarray, half_window: int) -> np.ndarray:
    """Move each candidate to the highest sample within +/- half_window."""
    refined = []
    for c in candidates:
        start = max(0, c - half_window)
        end = min(len(rs), c + half_window + 1)
        refined.append(start + np.argmax(rs[start:end]))
    return np.array(refined, dtype=int)


def enforce_min_rr(beats: np.ndarray, min_samples: int) -> np.ndarray:
    """Drop beats that follow the previous accepted beat too closely."""
    if len(beats) < 2 or min_samples <= 0:
        return beats
    kept = [beats[0]]
    for b in beats[1:]:
        if b - kept[-1] >= min_samples:
            kept.append(b)
    return np.array(kept, dtype=int)


def detect_beats(rs: np.ndarray, cfg: DetectorConfig):
    """Find beats in a range signal.

    Returns (beat sample indices, indices of samples marked as noisy).
    """
    threshold, envelope = dynamic_threshold(rs, cfg)
    noisy = np.where(envelope <= cfg.noise_threshold)[0]

    # check the threshold only every fs/25 samples (40 ms at 250 Hz)
    stride = int(cfg.target_fs / 25)
    above = rs > threshold
    above = np.repeat(above[::stride], stride)[: len(rs)]
    candidates = np.where(above[1:] & ~above[:-1])[0] + 1

    beats = refine_peaks(rs, candidates, int(cfg.refine_window * cfg.target_fs))
    beats = enforce_min_rr(beats, int(cfg.min_rr_interval * cfg.target_fs))
    return beats, noisy


def locate_r_peaks(filtered: np.ndarray, beats: np.ndarray, before: int, after: int) -> np.ndarray:
    """Move ECG beats from the range signal peak to the R peak.

    The range signal peaks a bit after the QRS complex (it looks at the
    previous 200 ms), and for wide or inverted complexes the delay can be
    over 50 ms. Here we take the largest absolute deflection of the
    baseline-corrected ECG in [beat - before, beat + after].
    """
    if len(beats) == 0:
        return beats
    magnitude = np.abs(filtered)
    located = [max(0, b - before) + np.argmax(magnitude[max(0, b - before): b + after + 1])
               for b in beats]
    return np.unique(np.array(located, dtype=int))
