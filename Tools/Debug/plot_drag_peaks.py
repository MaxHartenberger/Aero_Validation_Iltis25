import argparse
from pathlib import Path
import sys

# Ensure repo root on sys.path for package imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from aero_validation.io import load_data, ensure_columns
from aero_validation.segmentation import detect_drag_peaks
from aero_validation.plots import plot_peaks_overview


# CLI: parse arguments
parser = argparse.ArgumentParser(description="Plot velocity, steering, and braking; mark drag peaks")
parser.add_argument("--csv", type=Path, default=None, help="Input CSV path (default: Outputs/01_Data_Preparation/Aero_Validation_Signals.csv if present)")
parser.add_argument("--out", type=Path, default=None, help="Output PNG (default: Outputs/02_Run_Extraction/plots/drag_peaks_overview.png)")
parser.add_argument("--min-peak-speed", type=float, default=1.0, help="Minimum peak speed [m/s]")
parser.add_argument("--peak-window-s", type=float, default=0.5, help="Peak locality window [s]")
parser.add_argument("--peak-prominence", type=float, default=0.1, help="Peak prominence vs. local window [m/s]")
args = parser.parse_args()

# Load and check required columns
df = load_data(args.csv)
ensure_columns(df, ["t_s", "GPS1_VEL_MAG_ms"])

t = df["t_s"].to_numpy()
v = df["GPS1_VEL_MAG_ms"].to_numpy()
steer = df["ICU_SteeringAngle_deg"].to_numpy() if "ICU_SteeringAngle_deg" in df.columns else None
brake_f = df["ICU_Brakpresf_bar"].to_numpy() if "ICU_Brakpresf_bar" in df.columns else None
brake_r = df["ICU_Brakpresr_bar"].to_numpy() if "ICU_Brakpresr_bar" in df.columns else None

kwargs = dict(
    min_peak_speed=args.min_peak_speed,
    peak_window_s=args.peak_window_s,
    peak_prominence=args.peak_prominence,
)
peaks = detect_drag_peaks(t, v, config=kwargs)

repo = Path(__file__).resolve().parents[1]
out_png = Path(args.out) if args.out else (repo / "Outputs" / "02_Run_Extraction" / "plots" / "drag_peaks_overview.png")
out_png.parent.mkdir(parents=True, exist_ok=True)

plot_peaks_overview(t, v, out_png, steer=steer, brake_f=brake_f, brake_r=brake_r, peaks=peaks)
print(f"Plotted {len(peaks)} peaks to {out_png}")
