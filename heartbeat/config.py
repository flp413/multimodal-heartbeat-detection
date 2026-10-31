from dataclasses import dataclass, replace


@dataclass
class DetectorConfig:
    """All tunable parameters of the detector. Times are in seconds."""

    target_fs: int = 250        # every channel is resampled to this rate
    alpha: float = 0.25         # trimming proportion of the TMA filter

    # TMA baseline window (full width)
    window_ecg: float = 0.2
    window_bp: float = 1.0

    # range filter window
    length_ecg: float = 0.2
    length_bp: float = 0.4

    # dynamic threshold
    envelope_window: float = 1.0
    smoothing_samples: int = 80
    noise_threshold: float = 0.40

    # beat validation
    refine_window: float = 0.04  # search window around a candidate (+/-)
    min_rr_interval: float = 0.25

    # channel quality check (first N seconds of each channel)
    quality_window: int = 30
    rr_ratio_low: float = 0.8
    rr_ratio_high: float = 1.2
    min_quality_score: float = 0.80

    # multi-channel fusion
    merge_tolerance: float = 0.15
    min_agreeing_channels: int = 2
    median_filter_size: int = 19


# Settings before my changes, closer to the original paper. Used to compare
# both versions on the same records (see scripts/evaluate.py --preset).
INITIAL = DetectorConfig(
    refine_window=0.05,
    min_rr_interval=0.0,
    min_agreeing_channels=1,
)

IMPROVED = DetectorConfig()

PRESETS = {
    "improved": IMPROVED,
    "initial": INITIAL,
}


def get_preset(name: str) -> DetectorConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset '{name}', choose from {list(PRESETS)}")
    return replace(PRESETS[name])
