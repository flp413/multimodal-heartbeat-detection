from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .config import DetectorConfig
from .quality import ECG_NAMES

PULSATILE_NAMES = ["abp", "art", "bp", "pap", "pleth", "ppg", "pressure"]


@dataclass
class ChannelDetection:
    channel_idx: int
    signal_type: str
    beats: np.ndarray
    noisy: np.ndarray
    quality: float
    name: str = ""

    def clean_mask(self) -> np.ndarray:
        return ~np.isin(self.beats, self.noisy)


def _nearest(reference: np.ndarray, beats: np.ndarray) -> np.ndarray:
    if len(reference) == 1:
        return np.full_like(beats, reference[0])
    idx = np.clip(np.searchsorted(reference, beats), 1, len(reference) - 1)
    left = reference[idx - 1]
    right = reference[idx]
    return np.where(np.abs(beats - left) < np.abs(beats - right), left, right)


def correct_delay(reference: np.ndarray, beats: np.ndarray, kernel: int,
                  max_delay: Optional[int] = None) -> np.ndarray:
    """Shift beats of a channel onto the reference channel.

    The delay to the nearest reference beat is smoothed with a median filter,
    so single wrong matches don't move the result. BP channels lag behind the
    ECG by the pulse transit time, and that lag changes slowly over time.

    If max_delay is given, delays larger than that (e.g. when the reference
    channel has a gap) are not trusted and get interpolated from neighbours.
    """
    if len(beats) == 0 or len(reference) == 0:
        return beats

    delays = (beats - _nearest(reference, beats)).astype(float)
    if max_delay is not None:
        valid = np.abs(delays) <= max_delay
        if valid.sum() < 2:
            return beats
        smooth = medfilt(delays[valid], kernel_size=kernel)
        delays = np.interp(beats, beats[valid], smooth)
    else:
        delays = medfilt(delays, kernel_size=kernel)

    return beats - np.round(delays).astype(int)


def _pick(detections, signal_type, is_named) -> Optional[ChannelDetection]:
    """Choose the channel for a role (ECG or pulse).

    A channel whose name says what it is wins, even if the quality check
    preferred the other settings for it (some pressure signals look more
    regular with the ECG settings). Otherwise the best channel of that type.
    """
    named = [d for d in detections if is_named(d.name.lower())]
    candidates = named or [d for d in detections if d.signal_type == signal_type]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.quality, -d.channel_idx))


def fuse_vote(detections: List[ChannelDetection], cfg: DetectorConfig) -> np.ndarray:
    """Group beats from all channels and keep groups seen in enough clean channels."""
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
    return np.array(fused, dtype=int)


def fuse_primary(ecg: ChannelDetection, pulse: ChannelDetection, cfg: DetectorConfig) -> np.ndarray:
    """ECG-led fusion.

    Clean ECG beats are kept as they are, because their timing is the most
    precise. The pulsatile channel is only used where the ECG has a gap
    longer than gap_factor times the local RR interval (missed beats, noise,
    a disconnected lead). Its beats are shifted by the pulse transit time
    first, so they land where the R peak would have been.
    """
    beats = ecg.beats[ecg.clean_mask()]
    pulse_beats = pulse.beats[pulse.clean_mask()]
    if len(pulse_beats) == 0:
        return beats
    if len(beats) < 3:
        return beats if len(beats) else pulse_beats

    max_delay = int(cfg.max_delay * cfg.target_fs)
    shifted = correct_delay(ecg.beats, pulse_beats, cfg.median_filter_size, max_delay)

    margin = int(cfg.merge_tolerance * cfg.target_fs)
    rr = np.diff(beats)
    local_rr = median_filter(rr, size=15, mode="nearest")

    fill = [shifted[shifted < beats[0] - margin], shifted[shifted > beats[-1] + margin]]
    for i in np.where(rr > cfg.gap_factor * local_rr)[0]:
        fill.append(shifted[(shifted > beats[i] + margin) & (shifted < beats[i + 1] - margin)])

    return np.unique(np.concatenate([beats] + fill)).astype(int)


def fuse(detections: List[ChannelDetection], original_fs: float, cfg: DetectorConfig) -> np.ndarray:
    """Merge per-channel beats into one list, in samples at original_fs."""
    scale = original_fs / cfg.target_fs

    if not detections:
        return np.array([], dtype=int)

    if len(detections) == 1:
        d = detections[0]
        fused = d.beats[d.clean_mask()]
    elif cfg.fusion == "primary":
        ecg = _pick(detections, "ecg", lambda n: n in ECG_NAMES)
        pulse = _pick(detections, "bp", lambda n: any(k in n for k in PULSATILE_NAMES))
        if pulse is ecg:
            pulse = None
        if ecg is None or pulse is None:
            only = ecg or pulse
            fused = only.beats[only.clean_mask()]
        else:
            fused = fuse_primary(ecg, pulse, cfg)
    elif cfg.fusion == "vote":
        fused = fuse_vote(detections, cfg)
    else:
        raise ValueError(f"unknown fusion '{cfg.fusion}'")

    fused = np.round(np.asarray(fused) * scale).astype(int)
    return fused[fused >= 0]
