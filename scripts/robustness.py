"""Check what happens when the ECG drops out.

Parts of the ECG channel are replaced (flat line = disconnected lead, or
white noise) and the detector is run again. Compares the ECG alone with
the ECG + pulse fusion.

    python scripts/robustness.py data/set-p
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import wfdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import list_records  # noqa: E402

from heartbeat import HeartbeatDetector, get_preset  # noqa: E402
from heartbeat.evaluation import match_beats, scores  # noqa: E402
from heartbeat.io import read_reference  # noqa: E402

# (preset, keep pulse channel?)
METHODS = {
    "ECG only": ("improved", False),
    "ECG + BP, voting": ("voting", True),
    "ECG + BP, improved": ("improved", True),
}


def corrupt(signal, fs, fraction, kind, rng, skip=30, segment=20):
    """Replace about `fraction` of the signal after the first `skip` seconds."""
    out = signal.copy()
    seg = int(segment * fs)
    start_min = int(skip * fs)
    n_segments = int(round(fraction * (len(signal) - start_min) / seg))
    slots = (len(signal) - start_min) // seg
    for k in rng.choice(slots, size=min(n_segments, slots), replace=False):
        a = start_min + k * seg
        if kind == "flat":
            out[a:a + seg] = np.nanmedian(signal)
        else:
            out[a:a + seg] = rng.normal(0, np.nanstd(signal), len(out[a:a + seg]))
    return out


def run_one(job):
    data_dir, name, kind, fraction = job
    path = os.path.join(data_dir, name)
    record = wfdb.rdrecord(path)
    reference = read_reference(path)
    tolerance = int(0.05 * record.fs)

    names = list(record.sig_name)
    if "ECG" not in names or not any("BP" in n for n in names):
        return None
    ecg = names.index("ECG")
    bp = next(i for i, n in enumerate(names) if "BP" in n)

    rng = np.random.default_rng(int(name))
    signals = record.p_signal[:, [ecg, bp]].copy()
    if fraction > 0:
        signals[:, 0] = corrupt(signals[:, 0], record.fs, fraction, kind, rng)

    row = {"record": name, "kind": kind, "fraction": fraction}
    for label, (preset, with_pulse) in METHODS.items():
        cols = [0, 1] if with_pulse else [0]
        beats = HeartbeatDetector(get_preset(preset)).detect(
            signals[:, cols], record.fs, ["ECG", "BP"][: len(cols)])
        row[label] = match_beats(beats, reference, tolerance)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    parser.add_argument("--out", default="results/robustness.csv")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    records = list_records(args.data_dir)
    settings = [("flat", 0.0)] + [(k, f) for k in ("flat", "noise") for f in (0.1, 0.3, 0.5)]
    jobs = [(args.data_dir, r, k, f) for k, f in settings for r in records]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [r for r in pool.map(run_one, jobs) if r]

    summary = []
    for kind, fraction in settings:
        subset = [r for r in rows if r["kind"] == kind and r["fraction"] == fraction]
        line = {"corruption": "none" if fraction == 0 else kind,
                "fraction": fraction, "records": len(subset)}
        for label in METHODS:
            total = np.sum([r[label] for r in subset], axis=0)
            s = scores(*total)
            line[f"{label} Se"] = round(s["Se"], 2)
            line[f"{label} +P"] = round(s["+P"], 2)
        summary.append(line)
        print(line)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
