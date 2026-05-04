import argparse
from typing import Optional
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

# Ensure repo root on sys.path for package imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from aero_validation.io import load_data, ensure_columns, ensure_global_plots_dir, repo_root
from aero_validation.intervals import load_intervals_csv, load_intervals_txt
from aero_validation.plots import plot_overview, plot_run


# CLI
parser = argparse.ArgumentParser(description="Generate per-run outputs from exported intervals")
parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help=(
        "Input CSV path. If omitted, will auto-detect one of: "
        "Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv, "
        "Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned.csv, "
        "Outputs/01_Data_Preparation/Aero_Validation_Signals.csv, "
        "Data/Aero_Validation_Signals_cleaned.csv, Data/Aero_Validation_Signals.csv"
    ),
)
parser.add_argument("--intervals", type=Path, default=None, help="Intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
parser.add_argument("--outdir", type=Path, default=None, help="Output runs directory (default: Outputs/02_Run_Extraction/csv)")
parser.add_argument("--overview-out", type=Path, default=None, help="Overview plot output path (default: Outputs/02_Run_Extraction/plots/velocity_overview_with_runs.png)")
parser.add_argument("--steer-thresh", type=float, default=14.0, help="Cap steering subplot scale to this threshold [deg]")
args = parser.parse_args()

def _resolve_default_input_csv(repo: Path) -> Path | None:
    candidates = [
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals_cleaned_10ms.csv",
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals_cleaned.csv",
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals.csv",
        repo / "Data" / "Aero_Validation_Signals_cleaned.csv",
        repo / "Data" / "Aero_Validation_Signals.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


repo = repo_root()
csv_eff = Path(args.csv) if args.csv is not None else _resolve_default_input_csv(repo)
if csv_eff is None or not Path(csv_eff).exists():
    raise FileNotFoundError(
        "No input CSV found. Provide --csv, or generate one via: "
        "Code/01_Data_Preparation/build_cleaned_signals_10ms.py (writes Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned_10ms.csv)."
    )
df = load_data(csv_eff)
ensure_columns(df, ["t_s", "GPS1_VEL_MAG_ms"])

t = df["t_s"].to_numpy()
def _interp_to_t(t_ref: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interpolate series (ts, ys) onto t_ref without extrapolation (NaN outside)."""
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if ts.size == 0 or ys.size == 0:
        return np.full_like(t_ref, np.nan, dtype=float)
    # unique, sorted ts
    ts_u, idx = np.unique(ts, return_index=True)
    ys_u = ys[idx]
    out = np.full_like(t_ref, np.nan, dtype=float)
    mask = (t_ref >= ts_u[0]) & (t_ref <= ts_u[-1])
    if np.any(mask):
        out[mask] = np.interp(t_ref[mask], ts_u, ys_u)
    return out

def _load_channel_from_signals(repo: Path, file_base: str) -> Optional[np.ndarray]:
    """Try to load a per-channel CSV from Outputs/01_Data_Preparation/Signals and return values interpolated to df's t.

    file_base: basename without .csv, e.g. 'SBG_IMU_ACCEL_X'.
    """
    chan_dir = repo / "Outputs" / "01_Data_Preparation" / "Signals"
    csv_path = chan_dir / f"{file_base}.csv"
    if not csv_path.exists():
        return None
    try:
        tmp = pd.read_csv(csv_path)
        # Accept common time column names
        time_col_name = next((c for c in ("time", "t_s", "time [s]") if c in tmp.columns), None)
        if tmp.shape[1] < 2 or time_col_name is None:
            return None
        time_col = np.asarray(tmp[time_col_name], dtype=float)

        # Prefer an exact match to the channel name if present; otherwise pick the first non-time column.
        if file_base in tmp.columns and file_base != time_col_name:
            val_col_name = file_base
        else:
            val_col_name = next((c for c in tmp.columns if c != time_col_name), None)
        if val_col_name is None:
            return None
        vals = np.asarray(tmp[val_col_name], dtype=float)
        return _interp_to_t(t, time_col, vals)
    except Exception:
        return None

runs_root = Path(args.outdir) if args.outdir else (repo / "Outputs" / "02_Run_Extraction" / "csv")
intervals_default_csv = repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv"

if args.intervals:
    ipath = Path(args.intervals)
    if ipath.suffix.lower() == ".txt":
        intervals, meta, run_ids = load_intervals_txt(ipath, max_len=len(t))
    else:
        intervals, meta, run_ids = load_intervals_csv(ipath)
else:
    intervals, meta, run_ids = load_intervals_csv(intervals_default_csv)

if meta.get("row_count") is not None and meta["row_count"] != len(df):
    print(f"Warning: row_count mismatch (intervals {meta['row_count']} vs data {len(df)}). Proceeding.")
if meta.get("source_basename") is not None:
    src_name = os.path.basename(str(csv_eff)) if csv_eff is not None else os.path.basename(str(repo / 'Outputs' / '01_Data_Preparation' / 'Aero_Validation_Signals.csv'))
    if meta["source_basename"] != src_name:
        print(f"Warning: source_basename mismatch (intervals {meta['source_basename']} vs data {src_name}). Proceeding.")

runs_dir = runs_root
runs_dir.mkdir(parents=True, exist_ok=True)
global_plots_dir = ensure_global_plots_dir()
per_run_plots_dir = global_plots_dir

v = df["GPS1_VEL_MAG_ms"].to_numpy()
steer = df["ICU_SteeringAngle_deg"].to_numpy() if "ICU_SteeringAngle_deg" in df.columns else None
w_kmh = df["WHEEL_SPEED_kmh"].to_numpy() if "WHEEL_SPEED_kmh" in df.columns else None

"""
Hand-picked acceleration-related columns from Signals.
Only include if present; order defines plotting priority/colors.
Extend the list here if more longitudinal/IMU acceleration signals are available.
"""
acc_defs = [
    ("IMU_ACCEL_LONG_ms2", "IMU_ACCEL_LONG_ms2", "IMU_ACCEL_LONG_ms2"),
]
acc_series_all = []
for label, df_col, file_base in acc_defs:
    if df_col in df.columns:
        acc_series_all.append((label, df[df_col].to_numpy()))
    else:
        arr = _load_channel_from_signals(repo, file_base)
        if arr is not None:
            acc_series_all.append((label, arr))

if len(acc_series_all) == 0:
    print("Note: No hand-picked longitudinal acceleration columns found in data or Outputs/01_Data_Preparation/Signals. Fallback accel (dv/dt) will be used where applicable.")
else:
    print(f"Selected acceleration columns: {[label for (label, _) in acc_series_all]}")

acc_ylim = None
if len(acc_series_all) > 0:
    # Compute a common symmetric y-limit across all selected acceleration series
    vals = []
    for _, arr in acc_series_all:
        arr_f = np.asarray(arr, dtype=float)
        finite = arr_f[np.isfinite(arr_f)]
        if finite.size:
            vals.append(finite)
    if vals:
        all_vals = np.concatenate(vals)
        q_low, q_high = np.percentile(all_vals, [1.0, 99.0])
        max_abs = float(max(abs(q_low), abs(q_high)))
        acc_ylim = (-max_abs, max_abs)

overview_path = Path(args.overview_out) if args.overview_out else global_plots_dir / "velocity_overview_with_runs.png"
plot_overview(t, v, intervals, overview_path, steer=steer, wheel_kmh=w_kmh, run_ids=run_ids, acc_series=acc_series_all)

for (i0, i1), rid in zip(intervals, run_ids):
    sl = slice(i0, i1)
    df_run = df.iloc[sl].copy()
    run_csv = runs_dir / f"run_{rid:03d}.csv"
    df_run.to_csv(run_csv, index=False)

    t0_run = float(t[i0])
    t1_run = float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1])
    tmin = t0_run - 0.5
    tmax = t1_run + 0.5
    i0_pad = int(np.searchsorted(t, tmin, side="left"))
    i1_pad = int(np.searchsorted(t, tmax, side="right"))
    sl_pad = slice(max(0, i0_pad), min(len(t), i1_pad))

    t_seg = t[sl_pad]
    v_seg = v[sl_pad]
    steer_seg = steer[sl_pad] if steer is not None else None
    brake_f_seg = None
    brake_r_seg = None
    w_kmh_seg = w_kmh[sl_pad] if w_kmh is not None else None
    acc_series_seg = [(label, arr[sl_pad]) for (label, arr) in acc_series_all]
    plot_path = per_run_plots_dir / f"run_{rid:03d}_velocity_steering_brake.png"

    plot_run(
        t_seg,
        v_seg,
        steer_seg,
        brake_f_seg,
        brake_r_seg,
        plot_path,
        run_start=(t0_run, float(v[i0])),
        run_end=(t1_run, float(v[i1 - 1] if i1 - 1 < len(v) else v[-1])),
        steer_thresh=args.steer_thresh,
        w_kmh=w_kmh_seg,
        acc_series=acc_series_seg,
        acc_ylim=acc_ylim,
    )

print(f"Wrote {len(intervals)} runs to {runs_dir}")
print(f"Overview plot: {overview_path}")
