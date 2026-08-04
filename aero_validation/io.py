from __future__ import annotations

"""Data I/O and project path utilities.

This module centralizes repository path resolution, default data locations,
loading of CSV data with sanitization and derived signals, and column presence
checks used across the project.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .signals import sanitize_columns, add_derived_signals

__all__ = [
    "repo_root",
    "data_path",
    "ensure_global_plots_dir",
    "load_data",
    "ensure_columns",
]


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    The repo root is defined relative to this module's location, two levels up
    from the package directory.
    """
    return Path(__file__).resolve().parents[1]


def data_path() -> Path:
    """Return the default input CSV path for the validation dataset.

    Prefers the most processed, most likely-to-exist dataset first (e.g. the
    cleaned 10 ms output), then falls back to less processed / legacy paths.
    """
    repo = repo_root()
    candidates = [
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals_cleaned_10ms.csv",
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals_cleaned.csv",
        repo / "Outputs" / "01_Data_Preparation" / "Aero_Validation_Signals.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No input CSV found. Looked in:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nGenerate the dataset with: "
        "Code/01_Data_Preparation/build_cleaned_signals_10ms.py"
    )


def ensure_global_plots_dir() -> Path:
    """Ensure and return the top-level Plots directory.

    Creates ``Outputs/02_Run_Extraction/plots/`` if it does not exist.
    """
    p = repo_root() / "Outputs" / "02_Run_Extraction" / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_data(csv_path: Path | None = None, encoding: str = "utf-8") -> pd.DataFrame:
    """Load the dataset CSV, sanitize column names, and add derived signals.

    Args:
        csv_path: Optional path to the CSV file. If not provided, defaults to
            `data_path()`.
        encoding: Text encoding for the CSV. Defaults to 'utf-8'.

    Returns:
        A pandas DataFrame with sanitized column names and additional derived
        signals such as `WHEEL_SPEED_kmh`, `GPS1_VEL_MAG_ms`, and
        `IMU_ACCEL_LONG_ms2` when inputs permit.
    """
    csv = Path(csv_path) if csv_path else data_path()
    if not csv.exists():
        default = data_path()
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv}. "
            f"Provide an explicit path, or generate the cleaned dataset at: "
            f"{repo_root() / 'Outputs' / '01_Data_Preparation' / 'Aero_Validation_Signals_cleaned_10ms.csv'}. "
            f"Auto-detection default currently points to: {default}."
        )
    df = pd.read_csv(csv, encoding=encoding, engine="python", on_bad_lines="skip")
    df = sanitize_columns(df)
    df = add_derived_signals(df)
    return df


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a KeyError if any of the required columns are missing.

    Args:
        df: DataFrame to check for columns.
        required: Iterable of column names that must be present.

    Raises:
        KeyError: If any required column is not found in `df`.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
