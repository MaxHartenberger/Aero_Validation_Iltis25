from __future__ import annotations

"""Intervals utilities: reading interval CSV and parsing run ranges.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

__all__ = ["load_intervals_csv", "parse_ranges"]


def load_intervals_csv(path: Path) -> "tuple[list[tuple[int,int]], dict, list[int]]":
    """Load intervals from a CSV file produced by export scripts.

    Expects columns `start_idx`, `end_idx`, optionally `run_id`, and metadata
    columns `source_basename`, `row_count`.

    Args:
        path: Path to the intervals CSV.

    Returns:
        Tuple `(intervals, meta, run_ids)` where:
        - `intervals` is a list of `(start_idx, end_idx)` pairs
        - `meta` is a dict with `source_basename` and `row_count` if present
        - `run_ids` is a list of IDs (from CSV if present, otherwise sequential)
    """
    df = pd.read_csv(path)
    intervals = [(int(r.start_idx), int(r.end_idx)) for r in df.itertuples(index=False)]
    if "run_id" in df.columns:
        run_ids = [int(x) for x in df["run_id"].tolist()]
    else:
        run_ids = list(range(1, len(intervals) + 1))
    meta = dict(
        source_basename=str(df["source_basename"].iloc[0]) if "source_basename" in df.columns and len(df) > 0 else None,
        row_count=int(df["row_count"].iloc[0]) if "row_count" in df.columns and len(df) > 0 else None,
    )
    return intervals, meta, run_ids


def parse_ranges(spec: str) -> List[int]:
    """Parse a comma-separated ranges spec into a sorted list of unique IDs.

    Examples of specs: "2-16", "18-30,32-44,45-54", or individual IDs.

    Args:
        spec: Comma-separated string of ranges or IDs.

    Returns:
        Sorted list of unique integer IDs parsed from the spec.

    Raises:
        ValueError: If a part cannot be parsed as an integer range or ID.
    """
    ids: List[int] = []
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                raise ValueError(f"Invalid range '{p}'. Expected 'start-end'.")
            if end < start:
                start, end = end, start
            ids.extend(list(range(start, end + 1)))
        else:
            try:
                ids.append(int(p))
            except ValueError:
                raise ValueError(f"Invalid run id '{p}'. Expected integer.")
    return sorted(set(ids))
