# Results

All 100 records of PhysioNet/CinC Challenge 2014 `set-p`, 50 ms matching
window. "Average" is the mean of per-record values, "gross" counts all beats
together.

| Preset | Se avg | +P avg | F1 avg | Se gross | +P gross | F1 gross | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 99.73 | 95.16 | 97.19 | 99.73 | 94.32 | 96.95 | 72223 | 4353 | 192 |
| voting | 98.87 | 99.57 | 99.21 | 98.87 | 99.58 | 99.23 | 71599 | 299 | 816 |
| improved | 99.79 | 99.65 | 99.72 | 99.79 | 99.65 | 99.72 | 72262 | 257 | 153 |

Ablation (improved preset, one change at a time):

| Variant | Se | +P | F1 |
|---|---:|---:|---:|
| improved | 99.79 | 99.65 | 99.72 |
| without R peak localisation | 99.71 | 99.57 | 99.64 |
| without the ECG/pulse agreement check | 99.80 | 99.65 | 99.72 |

The agreement check doesn't change anything on the original data, it only
matters when the ECG is noisy (see `robustness.csv`: with 30 % of the ECG
replaced by noise, F1 goes from about 72 to 93).

Files:

- `metrics_<preset>.csv`: per-record TP, FP, FN, Se, +P, F1
- `robustness.csv`: gross Se / +P with part of the ECG replaced by a flat line
  or noise; only the ECG and BP channels are used there, so the 0 % row differs
  slightly from the tables above
- `annotations/`: detected beats of the improved preset, `<record>.asc` (text,
  same format as `rdann`/`wrann` use) and `<record>.qrs` (WFDB)

Lowest F1 per preset: baseline 149 (68.0), voting 118 (88.2), improved 133 (96.6).
