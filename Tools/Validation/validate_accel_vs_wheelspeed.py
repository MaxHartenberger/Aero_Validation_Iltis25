from pathlib import Path
import sys

# Ensure repo root on sys.path for package imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from aero_validation.io import load_data, ensure_columns
from aero_validation.validation import validate_accel_vs_wheelspeed


repo = Path(__file__).resolve().parents[1]
df = load_data()
ensure_columns(df, ["t_s", "WHEEL_SPEED_kmh", "IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"])
out_dir = repo / "Plots" / "Validation"
validate_accel_vs_wheelspeed(df, out_dir)
print(f"Wrote validation outputs to {out_dir}")
