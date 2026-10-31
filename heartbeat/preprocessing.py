import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean
from wfdb.processing import resample_sig

from .config import DetectorConfig

# Rows processed at once by the TMA filter. Keeps memory low for the
# 1 s BP window on 10 minute records.
_CHUNK = 20000


def resample(signal: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return signal
    return resample_sig(signal, fs, target_fs)[0]


def trimmed_moving_average(signal: np.ndarray, half_window: int, alpha: float) -> np.ndarray:
    """Centered trimmed moving average, window = 2 * half_window + 1 samples."""
    padded = np.pad(signal, half_window, mode="edge")
    windows = sliding_window_view(padded, 2 * half_window + 1)

    out = np.empty(len(signal))
    for start in range(0, len(signal), _CHUNK):
        stop = start + _CHUNK
        out[start:stop] = trim_mean(windows[start:stop], alpha / 2, axis=1)
    return out


def range_signal(signal: np.ndarray, window: int) -> np.ndarray:
    """Local max minus local min over the previous `window` samples."""
    padded = np.pad(signal, (window, 0), mode="edge")
    windows = sliding_window_view(padded, window)[: len(signal)]
    return windows.max(axis=1) - windows.min(axis=1)


def standardize(signal: np.ndarray) -> np.ndarray:
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 1e-6:
        signal = signal / std
    return signal


def preprocess(signal: np.ndarray, fs: float, signal_type: str, cfg: DetectorConfig) -> np.ndarray:
    """Raw channel -> range signal at cfg.target_fs."""
    signal = resample(signal, fs, cfg.target_fs)

    tma_window = cfg.window_ecg if signal_type == "ecg" else cfg.window_bp
    half_window = int(round(tma_window * cfg.target_fs / 2))
    filtered = signal - trimmed_moving_average(signal, half_window, cfg.alpha)
    filtered = standardize(filtered)

    length = cfg.length_ecg if signal_type == "ecg" else cfg.length_bp
    return range_signal(filtered, int(round(length * cfg.target_fs)))
