"""Backward-compatibility shim for drag computation imports.

.. deprecated::
    The functions previously defined here have moved to
    :mod:`aero_validation.drag`. This module re-exports them so existing
    ``from common import ...`` statements continue to work. New code should
    import directly from ``aero_validation`` or ``aero_validation.drag``.
"""

# Re-export everything from the package's drag module.
# This file exists only for backward compatibility with scripts that use
#   from common import ...
# after appending the parent directory to sys.path.

from aero_validation.drag import (  # noqa: F401  (re-export)
    VehicleConstants,
    load_constants,
    assign_config,
    robust_linear_fit,
    robust_linear_fit_multi,
    prepare_run_slice,
    derive_drag_metrics,
    fd_table_from_summary,
    summarize_by_config,
)

from aero_validation.io import repo_root  # noqa: F401


def data_dir():
    """Return the Data/ directory path."""
    from pathlib import Path
    return repo_root() / "Data"


def outputs_drag_dir():
    """Return the Outputs/03_Drag_Calculation/ directory path."""
    from pathlib import Path
    return repo_root() / "Outputs" / "03_Drag_Calculation"
