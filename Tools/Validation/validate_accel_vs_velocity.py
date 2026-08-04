from pathlib import Path

from aero_validation.io import load_data, ensure_columns
from aero_validation.validation import validate_accel_vs_velocity


repo = Path(__file__).resolve().parents[1]
df = load_data()
ensure_columns(df, ["t_s", "GPS1_VEL_MAG_ms", "IMU_ACCEL_X_ms2", "IMU_ACCEL_Y_ms2", "IMU_ACCEL_Z_ms2"])
out_dir = repo / "Plots" / "Validation"
validate_accel_vs_velocity(df, out_dir)
print(f"Wrote validation outputs to {out_dir}")
