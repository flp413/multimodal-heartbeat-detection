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

    # move ECG beats onto the R peak (search window before/after the beat)
    locate_r_peak: bool = True
    r_peak_before: float = 0.10
    r_peak_after: float = 0.05

    # multi-channel fusion
    fusion: str = "primary"      # "primary" or "vote", see fusion.py
    merge_tolerance: float = 0.15
    median_filter_size: int = 19
    max_delay: float = 0.5       # largest believable ECG -> pulse delay
    gap_factor: float = 1.5      # gap (in local RR) that the pulse channel may fill
    agreement_window: float = 5.0  # windows where ECG and pulse are compared, 0 = off
    agreement_tolerance: float = 0.06
    min_agreement: float = 0.6     # share of ECG beats that need a pulse beat nearby
    min_pulse_regularity: float = 0.5
    regularity_margin: float = 0.2   # pulse has to be this much more regular than the ECG
    min_agreeing_channels: int = 2  # only used by "vote"


# Close to the original paper: every clean beat from any channel is kept.
BASELINE = DetectorConfig(
    refine_window=0.05,
    min_rr_interval=0.0,
    locate_r_peak=False,
    fusion="vote",
    min_agreeing_channels=1,
)

# First round of changes: tighter refinement, minimum RR and a beat has to
# be seen in at least two channels.
VOTING = DetectorConfig(
    locate_r_peak=False,
    fusion="vote",
    min_agreeing_channels=2,
)

# Current version: R peak localisation and ECG-led fusion.
IMPROVED = DetectorConfig()

PRESETS = {
    "improved": IMPROVED,
    "voting": VOTING,
    "baseline": BASELINE,
}


def get_preset(name: str) -> DetectorConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset '{name}', choose from {list(PRESETS)}")
    return replace(PRESETS[name])
