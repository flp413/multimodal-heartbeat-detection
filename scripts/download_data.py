"""Download the PhysioNet/CinC Challenge 2014 training set (set-p) into data/set-p."""
import os

import wfdb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "set-p")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    wfdb.dl_database("challenge-2014/set-p", dl_dir=DATA_DIR)
    print(f"done, files are in {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
