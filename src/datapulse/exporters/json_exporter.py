"""JSON exporter: machine-readable report summary."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from datapulse.report import Report


def export_json(report: Report, path: Union[str, Path]) -> Path:
    """Serialize the full report to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(p)
    return p
