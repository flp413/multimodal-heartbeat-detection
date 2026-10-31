import os

import numpy as np
import wfdb


def write_asc(beats: np.ndarray, path: str) -> None:
    """Write beats in the ASCII format accepted by WFDB `wrann`."""
    with open(path, "w") as f:
        for b in beats:
            f.write(f"0:00:00.00 {b} N 0 0 0\n")


def write_qrs(beats: np.ndarray, record_name: str, fs: float, out_dir: str) -> None:
    """Write beats as a WFDB annotation file <record_name>.qrs."""
    wfdb.wrann(
        record_name,
        "qrs",
        sample=np.asarray(beats, dtype=int),
        symbol=["N"] * len(beats),
        fs=fs,
        write_dir=out_dir,
    )


def read_reference(record_path: str) -> np.ndarray:
    """Manually annotated beats (.atr) of a record."""
    return wfdb.rdann(record_path, "atr").sample


def record_name(record_path: str) -> str:
    return os.path.basename(record_path)
