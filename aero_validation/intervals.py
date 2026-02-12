from __future__ import annotations

"""Intervals utilities: reading interval CSV/TXT and parsing run ranges.
"""

from pathlib import Path
from typing import List, Tuple

import ast
import re
import pandas as pd


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


def load_intervals_txt(path: Path, max_len: int | None = None) -> "tuple[list[tuple[int,int]], dict, list[int]]":
    """Load intervals from a TXT file containing tuples or a Python list.

    Supports either one tuple per line `(start_idx, end_idx)` or a single list
    like `[(i0, i1), (i2, i3), ...]`. Strips comments starting with `#`.

    Args:
        path: Path to the TXT file.
        max_len: Optional maximum index to clamp to (e.g., length of time array).

    Returns:
        Tuple `(intervals, meta, run_ids)` similar to CSV loader.
    """
    text = Path(path).read_text(encoding="utf-8")
    no_comments = re.sub(r"[ \t]*#.*", "", text)
    lines = no_comments.splitlines()

    pairs: list[tuple[int, int]] = []
    if "[" in no_comments and "]" in no_comments:
        obj = ast.literal_eval(no_comments)
        if isinstance(obj, (list, tuple)):
            for item in obj:
                try:
                    i0 = int(item[0])
                    i1 = int(item[1])
                    pairs.append((i0, i1))
                except Exception:
                    continue
    else:
        pat = re.compile(r"^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                continue
            i0 = int(m.group(1))
            i1 = int(m.group(2))
            pairs.append((i0, i1))

    intervals: list[tuple[int, int]] = []
    for (i0, i1) in pairs:
        if max_len is not None:
            i0 = max(0, min(i0, max_len))
            i1 = max(0, min(i1, max_len))
        if i1 > i0:
            intervals.append((i0, i1))
    meta = dict(source_basename=None, row_count=None)
    run_ids = list(range(1, len(intervals) + 1))
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
