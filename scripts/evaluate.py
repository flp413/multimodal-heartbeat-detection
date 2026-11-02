"""Run the detector on every record in a folder and compare with the .atr annotations.

    python scripts/evaluate.py data/set-p
    python scripts/evaluate.py data/set-p --preset baseline
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import wfdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import HeartbeatDetector, get_preset  # noqa: E402
from heartbeat.config import PRESETS  # noqa: E402
from heartbeat.evaluation import match_beats, scores  # noqa: E402
from heartbeat.io import read_reference, write_asc, write_qrs  # noqa: E402


def list_records(data_dir):
    records_file = os.path.join(data_dir, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file) as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea"))


def run_one(job):
    data_dir, name, preset, tolerance_sec, ann_dir = job
    path = os.path.join(data_dir, name)
    fs = wfdb.rdheader(path).fs

    beats = HeartbeatDetector(get_preset(preset)).detect_record(path)
    if ann_dir:
        write_asc(beats, os.path.join(ann_dir, f"{name}.asc"))
        write_qrs(beats, name, fs, ann_dir)

    tp, fp, fn = match_beats(beats, read_reference(path), int(tolerance_sec * fs))
    return {"record": name, "TP": tp, "FP": fp, "FN": fn, **scores(tp, fp, fn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    parser.add_argument("--preset", default="improved", choices=list(PRESETS))
    parser.add_argument("--tolerance", type=float, default=0.05, help="matching window in seconds")
    parser.add_argument("--out", default=None, help="CSV with per-record results")
    parser.add_argument("--annotations", default=None, help="folder to also save .asc and .qrs files")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    out = args.out or f"results/metrics_{args.preset}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.annotations:
        os.makedirs(args.annotations, exist_ok=True)

    jobs = [(args.data_dir, r, args.preset, args.tolerance, args.annotations)
            for r in list_records(args.data_dir)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run_one, jobs))

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})

    tp = sum(r["TP"] for r in rows)
    fp = sum(r["FP"] for r in rows)
    fn = sum(r["FN"] for r in rows)
    gross = scores(tp, fp, fn)
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("Se", "+P", "F1")}

    print(f"{len(rows)} records, preset '{args.preset}', tolerance {args.tolerance * 1000:.0f} ms")
    print(f"  average over records: Se {mean['Se']:.2f}  +P {mean['+P']:.2f}  F1 {mean['F1']:.2f}")
    print(f"  gross (all beats):    Se {gross['Se']:.2f}  +P {gross['+P']:.2f}  F1 {gross['F1']:.2f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
