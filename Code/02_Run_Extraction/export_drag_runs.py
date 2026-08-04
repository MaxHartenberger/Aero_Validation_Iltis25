import argparse
from pathlib import Path
import os

import pandas as pd

from aero_validation.io import load_data, ensure_columns, data_path, repo_root, ensure_global_plots_dir
from aero_validation.plots import plot_overview
from aero_validation.segmentation import segment_drag_runs, segment_drag_runs_with_debug


# CLI
parser = argparse.ArgumentParser(description="Export straight-line drag run intervals from a dataset")
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
parser.add_argument("--out", type=Path, default=None, help="Output intervals CSV (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv)")
parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing for peaks and segmentation")
parser.add_argument("--debug-out", type=Path, default=None, help="Write a debug CSV with acceptance/rejection reasons per peak")
# Optional thresholds
parser.add_argument("--steer-thresh", type=float, default=10.0, help="Steering threshold [deg]")
parser.add_argument("--straight-ratio", type=float, default=0.75, help="Minimum fraction of samples under steering threshold")
parser.add_argument("--min-peak-speed", type=float, default=15.0, help="Minimum peak speed [m/s]")
parser.add_argument("--peak-window-s", type=float, default=0.5, help="Peak locality window [s]")
parser.add_argument("--peak-prominence", type=float, default=0.05, help="Peak prominence vs. local window [m/s]")
parser.add_argument("--reaccel-eps", type=float, default=0.2, help="Re-acceleration epsilon [m/s]")
parser.add_argument("--accel-window-s", type=float, default=0.5, help="Acceleration window [s]")
parser.add_argument("--brake-thresh", type=float, default=0.05, help="Brake threshold [bar]")
parser.add_argument("--brake-spike-thresh", type=float, default=0.1, help="Brake spike threshold [bar/s]")
parser.add_argument("--turn-window-s", type=float, default=0.3, help="Turning sustain window [s]")
parser.add_argument("--min-dur-s", type=float, default=1.5, help="Minimum run duration [s]")
parser.add_argument("--max-dur-s", type=float, default=20.0, help="Maximum run duration [s]")
parser.add_argument("--stop-speed-th", type=float, default=0.2, help="Stop speed threshold [m/s]")
args = parser.parse_args()

kwargs = dict(
    steer_thresh=args.steer_thresh,
    straight_ratio=args.straight_ratio,
    min_peak_speed=args.min_peak_speed,
    peak_window_s=args.peak_window_s,
    peak_prominence=args.peak_prominence,
    reaccel_eps=args.reaccel_eps,
    accel_window_s=args.accel_window_s,
    brake_thresh=args.brake_thresh,
    brake_spike_thresh=args.brake_spike_thresh,
    turn_window_s=args.turn_window_s,
    min_dur_s=args.min_dur_s,
    max_dur_s=args.max_dur_s,
    stop_speed_th=args.stop_speed_th,
    smooth=(not args.no_smooth),
)

repo = repo_root()
csv_eff = Path(args.csv) if args.csv else data_path()
df = load_data(csv_eff)
ensure_columns(df, ["t_s", "GPS1_VEL_MAG_ms", "ICU_SteeringAngle_deg"])

t = df["t_s"].to_numpy()
v = df["GPS1_VEL_MAG_ms"].to_numpy()
steer = df["ICU_SteeringAngle_deg"].to_numpy()
brake_f = df["ICU_Brakpresf_bar"].to_numpy() if "ICU_Brakpresf_bar" in df.columns else None
brake_r = df["ICU_Brakpresr_bar"].to_numpy() if "ICU_Brakpresr_bar" in df.columns else None
w_kmh = df["WHEEL_SPEED_kmh"].to_numpy() if "WHEEL_SPEED_kmh" in df.columns else None

if args.debug_out is not None:
    intervals, debug_rows = segment_drag_runs_with_debug(t, v, steer, brake_f=brake_f, brake_r=brake_r, config=kwargs)
else:
    intervals = segment_drag_runs(t, v, steer, brake_f=brake_f, brake_r=brake_r, config=kwargs)

# Generate overview plot with runs
run_ids = list(range(1, len(intervals) + 1))
global_plots_dir = ensure_global_plots_dir()
overview_path = global_plots_dir / "velocity_overview_with_runs.png"
plot_overview(t, v, intervals, overview_path, steer=steer, wheel_kmh=w_kmh, run_ids=run_ids)

out_csv = Path(args.out) if args.out else (repo / "Outputs" / "02_Run_Extraction" / "csv" / "drag_intervals.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for rid, (i0, i1) in enumerate(intervals, start=1):
    t0 = float(t[i0])
    t1 = float(t[i1 - 1]) if i1 - 1 < len(t) else float(t[-1])
    rows.append(dict(
        run_id=rid,
        start_idx=i0,
        end_idx=i1,
        t0_s=t0,
        t1_s=t1,
        duration_s=float(t1 - t0),
        v_start_ms=float(v[i0]),
        v_end_ms=float(v[i1 - 1] if i1 - 1 < len(v) else v[-1]),
    ))

src_for_meta = csv_eff if csv_eff is not None else data_path()
meta = dict(
    source_basename=os.path.basename(str(src_for_meta)),
    row_count=int(len(df)),
)

intervals_df = pd.DataFrame(rows)
intervals_df["source_basename"] = meta["source_basename"]
intervals_df["row_count"] = meta["row_count"]
intervals_df.to_csv(out_csv, index=False)

if args.debug_out is not None:
    debug_path = Path(args.debug_out)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    dbg_df = pd.DataFrame(debug_rows)
    dbg_df.to_csv(debug_path, index=False)

print(f"Exported {len(intervals)} drag intervals to {out_csv}")
print(f"Overview plot: {overview_path}")
