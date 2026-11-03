"""Make the figures used in the README (figures/*.png).

    python scripts/make_figures.py data/set-p
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import DetectorConfig, HeartbeatDetector  # noqa: E402
from heartbeat.detection import detect_beats, dynamic_threshold, locate_r_peaks  # noqa: E402
from heartbeat.preprocessing import (  # noqa: E402
    remove_baseline,
    resample,
    to_range_signal,
    trimmed_moving_average,
)

OUT = "figures"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#e6e5e1", "#1baf7a"
COLORS = {"baseline": GRAY, "voting": ORANGE, "improved": BLUE}
LABELS = {
    "baseline": "baseline (paper settings)",
    "voting": "2-channel voting",
    "improved": "ECG-led fusion (final)",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.2,
})


def pipeline(data_dir, record="150", start=60.0, length=6.0):
    cfg = DetectorConfig()
    rec = wfdb.rdrecord(os.path.join(data_dir, record))
    ref = wfdb.rdann(os.path.join(data_dir, record), "atr").sample / rec.fs
    fs = cfg.target_fs

    ecg = resample(np.nan_to_num(rec.p_signal[:, 0]), rec.fs, fs)
    bp = resample(np.nan_to_num(rec.p_signal[:, 1]), rec.fs, fs)
    half = int(round(cfg.window_ecg * fs / 2))
    baseline = trimmed_moving_average(ecg, half, cfg.alpha)
    filtered = remove_baseline(ecg, fs, "ecg", cfg)
    rs = to_range_signal(filtered, "ecg", cfg)
    threshold, _ = dynamic_threshold(rs, cfg)
    beats, _ = detect_beats(rs, cfg)
    r_peaks = locate_r_peaks(filtered, beats, int(cfg.r_peak_before * fs), int(cfg.r_peak_after * fs))
    bp_beats, _ = detect_beats(to_range_signal(remove_baseline(bp, fs, "bp", cfg), "bp", cfg), cfg)

    a, b = int(start * fs), int((start + length) * fs)
    t = np.arange(a, b) / fs - start
    sel = lambda x: x[(x >= a) & (x < b)]  # noqa: E731
    refs = ref[(ref >= start) & (ref < start + length)] - start

    fig, axes = plt.subplots(4, 1, figsize=(8, 7.6), sharex=True, gridspec_kw={"hspace": 0.4})
    panels = [
        ("a) Raw ECG and TMA baseline", "mV"),
        ("b) Baseline removed", "z"),
        ("c) Range signal and threshold", "range"),
        ("d) Blood pressure channel", "mmHg"),
    ]
    axes[0].plot(t, ecg[a:b], color=INK, lw=0.9, label="ECG")
    axes[0].plot(t, baseline[a:b], color=ORANGE, lw=1.6, label="TMA baseline")
    axes[1].plot(t, filtered[a:b], color=INK, lw=0.9)
    dots = {"ms": 5, "color": BLUE, "mec": "white", "mew": 1}
    r_shown = sel(r_peaks)
    beats_shown = sel(beats)
    bp_shown = sel(bp_beats)
    axes[1].plot(r_shown / fs - start, filtered[r_shown], "o", label="R peak (final beat position)", **dots)
    axes[2].plot(t, rs[a:b], color=INK, lw=0.9, label="range signal")
    axes[2].plot(t, threshold[a:b], color=ORANGE, lw=1.6, label="dynamic threshold")
    axes[2].plot(beats_shown / fs - start, rs[beats_shown], "o", label="range peak", **dots)
    axes[3].plot(t, bp[a:b], color=INK, lw=0.9)
    axes[3].plot(bp_shown / fs - start, bp[bp_shown], "o", label="pulse beat", **dots)

    for ax, (title, unit) in zip(axes, panels):
        for r in refs:
            ax.axvline(r, color=REF, lw=1, alpha=0.6, zorder=0)
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel(unit)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncol=3, fontsize=8)
    axes[-1].set_xlabel(f"time (s), record {record} from {start:.0f} s; green lines = reference annotations")
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(OUT, "pipeline.png"))
    plt.close(fig)


def comparison():
    rows = []
    for preset in COLORS:
        df = pd.read_csv(f"results/metrics_{preset}.csv")
        rows.append((preset, df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})

    # average scores, dot plot (axis does not start at 0 on purpose, no bars)
    metrics = ["Se", "+P", "F1"]
    for i, (preset, df) in enumerate(rows):
        values = [df[m].mean() for m in metrics]
        y = np.arange(len(metrics)) + (i - 1) * 0.22
        ax1.plot(values, y, "o", ms=8, color=COLORS[preset], mec="white", mew=1.5, label=LABELS[preset])
        for v, yy in zip(values, y):
            ax1.annotate(f"{v:.2f}", (v, yy), xytext=(6, -3), textcoords="offset points",
                         fontsize=7.5, color=MUTED)
    ax1.set_yticks(range(len(metrics)), ["Se (%)", "+P (%)", "F1 (%)"])
    ax1.invert_yaxis()
    ax1.set_xlim(94.5, 100.6)
    ax1.set_title("Average over 100 records", loc="left")
    ax1.grid(axis="y", visible=False)

    # per-record F1, sorted from worst to best
    for preset, df in rows:
        f1 = np.sort(df["F1"].values)
        ax2.plot(np.arange(1, len(f1) + 1), f1, color=COLORS[preset], lw=2, label=LABELS[preset])
    ax2.set_ylim(70, 100.5)
    ax2.set_xlabel("records, sorted by F1")
    ax2.set_ylabel("F1 (%)")
    ax2.set_title("F1 per record", loc="left")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "comparison.png"))
    plt.close(fig)


def robustness():
    df = pd.read_csv("results/robustness.csv")
    methods = {"ECG only": AQUA, "ECG + BP, voting": ORANGE, "ECG + BP, improved": BLUE}
    clean = df[df["fraction"] == 0].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    titles = {"flat": "ECG lead disconnected (flat line)", "noise": "ECG covered by noise"}
    for ax, (kind, title) in zip(axes, titles.items()):
        part = pd.concat([clean.to_frame().T, df[df["corruption"] == kind]])
        x = part["fraction"].astype(float) * 100
        for label, color in methods.items():
            se, ppv = part[f"{label} Se"].astype(float), part[f"{label} +P"].astype(float)
            f1 = 2 * se * ppv / (se + ppv)
            ax.plot(x, f1, "-o", color=color, lw=2, ms=6, mec="white", mew=1.2, label=label)
            if label == "ECG + BP, voting":
                continue
            ax.annotate(f"{f1.iloc[-1]:.1f}", (x.iloc[-1], f1.iloc[-1]), xytext=(6, -3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
        ax.set_title(title, loc="left")
        ax.set_xlabel("share of the ECG replaced (%)")
        ax.set_xticks([0, 10, 30, 50])
        ax.set_xlim(-3, 58)
    axes[0].set_ylabel("F1 (%), all beats")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "robustness.png"))
    plt.close(fig)


def failure_case(data_dir, record="133", length=10.0):
    """Worst record of the final version, around the densest cluster of errors."""
    path = os.path.join(data_dir, record)
    rec = wfdb.rdrecord(path)
    fs = rec.fs
    ref = wfdb.rdann(path, "atr").sample
    beats = HeartbeatDetector(DetectorConfig()).detect_record(path)
    tol = int(0.05 * fs)
    fp = np.array([b for b in beats if np.min(np.abs(ref - b)) > tol])
    fn = np.array([r for r in ref if np.min(np.abs(beats - r)) > tol])
    errors = np.sort(np.concatenate([fp, fn]))
    win = int(length * fs)
    counts = [np.sum((errors >= e) & (errors < e + win)) for e in errors]
    a = max(0, int(errors[int(np.argmax(counts))] - 1 * fs))
    b = a + win
    t = np.arange(a, b) / fs
    sel = lambda x: x[(x >= a) & (x < b)]  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    for ax, ch, unit in zip(axes, [0, 1], ["mV", "mmHg"]):
        sig = rec.p_signal[:, ch]
        ax.plot(t, sig[a:b], color=INK, lw=0.9)
        for r in sel(ref):
            ax.axvline(r / fs, color=REF, lw=1, alpha=0.6, zorder=0)
        ax.set_ylabel(f"{rec.sig_name[ch]} ({unit})")
    ecg = rec.p_signal[:, 0]
    fp_shown = sel(fp)
    fn_shown = sel(fn)
    tp = np.setdiff1d(sel(beats), fp_shown)
    axes[0].plot(tp / fs, ecg[tp], "o", ms=5, color=BLUE, mec="white", mew=1, label="correct detection")
    if len(fp_shown):
        axes[0].plot(fp_shown / fs, ecg[fp_shown], "x", ms=8, mew=2, color=ORANGE, label="false positive")
    if len(fn_shown):
        axes[0].plot(fn_shown / fs, ecg[fn_shown], "s", ms=7, mfc="none", mew=1.5, color=INK,
                     label="missed beat")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8)
    axes[0].set_title(f"Record {record}: where the detector still fails", loc="left", pad=22)
    axes[1].set_xlabel("time (s); green lines = reference annotations")
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(OUT, f"failure_case_{record}.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pipeline(args.data_dir)
    comparison()
    if os.path.exists("results/robustness.csv"):
        robustness()
    worst = pd.read_csv("results/metrics_improved.csv").sort_values("F1").iloc[0]["record"]
    failure_case(args.data_dir, str(int(worst)))
    print(f"figures saved to {OUT}/")


if __name__ == "__main__":
    main()
