from typing import Optional

import numpy as np
import wfdb

from .config import DetectorConfig
from .detection import detect_beats, locate_r_peaks
from .fusion import ChannelDetection, fuse
from .preprocessing import remove_baseline, to_range_signal
from .quality import classify_channel


class HeartbeatDetector:
    """Detects heartbeats in a multi-channel WFDB record (ECG + pulsatile)."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()

    def detect_record(self, record_path: str) -> np.ndarray:
        record = wfdb.rdrecord(record_path)
        return self.detect(record.p_signal, record.fs, record.sig_name)

    def detect(self, signals: np.ndarray, fs: float, names) -> np.ndarray:
        """signals: (n_samples, n_channels). Returns beat positions in samples at fs."""
        detections = []
        for ch, name in enumerate(names):
            signal = np.nan_to_num(signals[:, ch])
            signal_type, score = classify_channel(signal, fs, name, self.cfg)
            if not signal_type:
                continue

            filtered = remove_baseline(signal, fs, signal_type, self.cfg)
            beats, noisy = detect_beats(to_range_signal(filtered, signal_type, self.cfg), self.cfg)
            if signal_type == "ecg" and self.cfg.locate_r_peak:
                beats = locate_r_peaks(
                    filtered, beats,
                    int(self.cfg.r_peak_before * self.cfg.target_fs),
                    int(self.cfg.r_peak_after * self.cfg.target_fs),
                )
            detections.append(ChannelDetection(ch, signal_type, beats, noisy, score, name))

        return fuse(detections, fs, self.cfg)
