"""Detect heartbeats in one WFDB record and write <record>.asc and <record>.qrs.

    python scripts/detect.py data/set-p/100
    python scripts/detect.py data/set-p/100 --preset voting
"""
import argparse
import os
import sys

import wfdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heartbeat import HeartbeatDetector, get_preset  # noqa: E402
from heartbeat.config import PRESETS  # noqa: E402
from heartbeat.io import record_name, write_asc, write_qrs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", help="path to the record, without extension")
    parser.add_argument("--output-dir", default="results/annotations")
    parser.add_argument("--preset", default="voting", choices=list(PRESETS))
    parser.add_argument("--no-qrs", action="store_true", help="only write the .asc file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    name = record_name(args.record)

    detector = HeartbeatDetector(get_preset(args.preset))
    beats = detector.detect_record(args.record)

    write_asc(beats, os.path.join(args.output_dir, f"{name}.asc"))
    if not args.no_qrs:
        fs = wfdb.rdheader(args.record).fs
        write_qrs(beats, name, fs, args.output_dir)

    print(f"{name}: {len(beats)} beats")


if __name__ == "__main__":
    main()
