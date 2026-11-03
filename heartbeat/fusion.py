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


def _regularity(beats: np.ndarray) -> float:
    """Share of consecutive RR ratios between 0.8 and 1.2."""
    if len(beats) < 3:
        return 0.0
    rr = np.diff(beats).astype(float)
    ratios = rr[1:] / np.maximum(rr[:-1], 1)
    return float(np.mean((ratios > 0.8) & (ratios < 1.2)))


def _disagreeing_windows(ecg_beats, pulse_beats, cfg):
    """Mark windows where the ECG beats have no pulse beat next to them.

    Noise on the ECG lead gives a lot of extra beats that the range signal
    can't tell apart from QRS complexes. In such a window most ECG beats
    don't line up with the (already shifted) pulse beats. A window is only
    handed over to the pulse if the pulse rhythm there is also clearly more
    regular than the ECG one, otherwise we can't tell which channel is wrong
    (e.g. an arrhythmia, or a pressure signal with a strong dicrotic notch).

    Returns a list of (start, end) sample ranges.
    """
    window = int(cfg.agreement_window * cfg.target_fs)
    tolerance = int(cfg.agreement_tolerance * cfg.target_fs)
    end = max(ecg_beats[-1], pulse_beats[-1]) + 1

    bad = []
    for start in range(0, end, window):
        e = ecg_beats[(ecg_beats >= start) & (ecg_beats < start + window)]
        p = pulse_beats[(pulse_beats >= start) & (pulse_beats < start + window)]
        if len(p) < 3 or _regularity(p) < cfg.min_pulse_regularity:
            continue
        if len(e):
            matched = np.min(np.abs(e[:, None] - p[None, :]), axis=1) <= tolerance
            if matched.mean() >= cfg.min_agreement:
                continue
            if len(e) >= 3 and _regularity(e) + cfg.regularity_margin >= _regularity(p):
                continue
        bad.append((start, start + window))
    return bad


def _in_ranges(beats, ranges):
    mask = np.zeros(len(beats), dtype=bool)
    for a, b in ranges:
        mask |= (beats >= a) & (beats < b)
    return mask


def pulse_delay(ecg_beats, pulse_beats, cfg, exclude=()):
    """Delay of every pulse beat behind the ECG (pulse transit time), in samples.

    Delays are measured to the nearest ECG beat. Values far from the
    overall median, and beats inside `exclude` ranges, are treated as wrong
    matches; the rest is median filtered and interpolated.
    """
    raw = (pulse_beats - _nearest(ecg_beats, pulse_beats)).astype(float)
    max_delay = cfg.max_delay * cfg.target_fs
    plausible = np.abs(raw) <= max_delay
    if plausible.sum() < 2:
        return np.zeros(len(pulse_beats))

    typical = np.median(raw[plausible])
    valid = plausible & (np.abs(raw - typical) <= cfg.merge_tolerance * cfg.target_fs)
    valid &= ~_in_ranges(pulse_beats, exclude)
    if valid.sum() < 2:
        return np.full(len(pulse_beats), typical)

    smooth = medfilt(raw[valid], kernel_size=cfg.median_filter_size)
    return np.interp(pulse_beats, pulse_beats[valid], smooth)


def fuse_primary(ecg: ChannelDetection, pulse: ChannelDetection, cfg: DetectorConfig) -> np.ndarray:
    """ECG-led fusion.

    Clean ECG beats are kept as they are, because their timing is the most
    precise. The pulse channel, shifted back by the pulse transit time, is
    used in two cases:
      * windows where the ECG beats don't agree with the pulse (noisy lead),
      * gaps in the ECG longer than gap_factor times the local RR interval
        (missed beats, disconnected lead, start/end of the record).
    """
    beats = ecg.beats[ecg.clean_mask()]
    pulse_beats = pulse.beats[pulse.clean_mask()]
    if len(pulse_beats) < 3:
        return beats
    if len(beats) < 3:
        # (almost) nothing usable on the ECG, the pulse is all we have
        return pulse_beats

    # first pass with a rough delay to find where the ECG is wrong,
    # second pass estimates the delay only from the parts where it is fine
    shifted = pulse_beats - np.round(pulse_delay(beats, pulse_beats, cfg)).astype(int)
    bad = _disagreeing_windows(beats, shifted, cfg) if cfg.agreement_window > 0 else []
    if bad:
        shifted = pulse_beats - np.round(pulse_delay(beats, pulse_beats, cfg, bad)).astype(int)
        replaced = _in_ranges(beats, bad)
        beats = np.unique(np.concatenate([beats[~replaced], shifted[_in_ranges(shifted, bad)]]))
        if len(beats) < 3:
            return beats

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
