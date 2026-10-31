from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import medfilt

from .config import DetectorConfig


@dataclass
class ChannelDetection:
    channel_idx: int
    signal_type: str
    beats: np.ndarray
    noisy: np.ndarray
    quality: float


def correct_delay(reference: np.ndarray, beats: np.ndarray, kernel: int) -> np.ndarray:
    """Shift beats of a channel onto the reference channel.

    The delay to the nearest reference beat is smoothed with a median filter,
    so single wrong matches don't move the result. BP channels lag behind the
    ECG by the pulse transit time, and that lag changes slowly over time.
    """
    if len(beats) == 0 or len(reference) == 0:
        return beats

    idx = np.clip(np.searchsorted(reference, beats), 1, len(reference) - 1)
    left = reference[idx - 1]
    right = reference[idx]
    nearest = np.where(np.abs(beats - left) < np.abs(beats - right), left, right)
    if len(reference) == 1:
        nearest = np.full_like(beats, reference[0])

    delays = medfilt((beats - nearest).astype(float), kernel_size=kernel)
    corrected = beats - delays.astype(int)
    return corrected[corrected >= 0]


def fuse(detections: List[ChannelDetection], original_fs: float, cfg: DetectorConfig) -> np.ndarray:
    """Merge per-channel beats into one list, in samples at original_fs."""
    scale = original_fs / cfg.target_fs

    if not detections:
        return np.array([], dtype=int)

    if len(detections) == 1:
        d = detections[0]
        clean = d.beats[~np.isin(d.beats, d.noisy)]
        return np.round(clean * scale).astype(int)

    reference = detections[0]
    events = []  # (time, channel, is_clean)
    for d in detections:
        beats = d.beats
        if d.channel_idx != reference.channel_idx:
            beats = correct_delay(reference.beats, beats, cfg.median_filter_size)
        clean = ~np.isin(beats, d.noisy)
        events.extend(zip(beats, [d.channel_idx] * len(beats), clean))
    events.sort(key=lambda e: e[0])

    tolerance = int(cfg.merge_tolerance * cfg.target_fs)
    fused = []
    i = 0
    while i < len(events):
        start_time = events[i][0]
        group = [events[i]]
        j = i + 1
        while j < len(events) and events[j][0] - start_time <= tolerance:
            if events[j][1] not in [g[1] for g in group]:
                group.append(events[j])
            j += 1

        clean_times = [t for t, _, ok in group if ok]
        if len(clean_times) >= cfg.min_agreeing_channels:
            fused.append(int(round(np.mean(clean_times))))
        i = j

    return np.round(np.array(fused, dtype=int) * scale).astype(int)
