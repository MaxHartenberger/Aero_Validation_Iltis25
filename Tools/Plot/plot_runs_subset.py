#!/usr/bin/env python3
"""
Plot subset overviews of drag run intervals for specified run ID ranges.

Defaults: generates 1–12, 13–24, 25–36, 37–46 into Outputs/02_Run_Extraction/plots/overview_runs_<range>.png
"""
from pathlib import Path
import argparse

import pandas as pd

from aero_validation.io import load_data
from aero_validation.intervals import load_intervals_csv, parse_ranges
from aero_validation.plots import plot_overview_runs_subset


# CLI args
ap = argparse.ArgumentParser(description="Plot subset overviews for selected run ID ranges.")
ap.add_argument('--intervals', type=str, default='Outputs/02_Run_Extraction/csv/drag_intervals.csv', help="Path to drag_intervals.csv (default: Outputs/02_Run_Extraction/csv/drag_intervals.csv).")
ap.add_argument('--ranges', type=str, default='1-12,13-24,25-36,37-46', help="Comma-separated ranges (default: '1-12,13-24,25-36,37-46').")
ap.add_argument('--data', type=str, default='Outputs/01_Data_Preparation/Aero_Validation_Signals_cleaned.csv', help="CSV with signals (default cleaned).")
ap.add_argument('--outdir', type=str, default='Outputs/02_Run_Extraction/plots', help="Output directory for plots (default: Outputs/02_Run_Extraction/plots).")
args = ap.parse_args()

intervals_path = Path(args.intervals)
out_dir = Path(args.outdir)
out_dir.mkdir(parents=True, exist_ok=True)

# Load signals
df = load_data(Path(args.data))
t = df['t_s'].to_numpy()
v = df['GPS1_VEL_MAG_ms'].to_numpy()
steer = df['ICU_SteeringAngle_deg'].to_numpy() if 'ICU_SteeringAngle_deg' in df.columns else None

intervals, meta, run_ids_all = load_intervals_csv(intervals_path)

# Generate plots for each specified range spec separately
range_specs = [s.strip() for s in args.ranges.split(',') if s.strip()]
any_written = False
for spec in range_specs:
    try:
        run_ids = parse_ranges(spec)
    except ValueError as e:
        print(f"Skipping invalid range '{spec}': {e}")
        continue
    available = set(run_ids_all)
    selected = sorted(set(run_ids) & available)
    out_path = out_dir / f"overview_runs_{spec.replace(' ','')}.png"
    plot_overview_runs_subset(t, v, intervals, selected, out_path, steer=steer, run_ids_all=run_ids_all)
    print(f"Wrote subset overview ({spec}): {out_path}")
    if len(selected) < len(run_ids):
        missing = sorted(set(run_ids) - available)
        if missing:
            print(f"  Missing run IDs in data for '{spec}': {missing}")
    any_written = True

if not any_written:
    print("No valid ranges provided; nothing written.")
