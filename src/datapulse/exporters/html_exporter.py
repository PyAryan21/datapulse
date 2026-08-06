"""HTML exporter: self-contained interactive report (Jinja2 template + inline Plotly.js)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import plotly
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from datapulse.report import Report

_TEMPLATE_DIR = Path(__file__).parent / ".." / "templates"


def _load_template(name: str):
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(name)


def _fig_to_json(fig: Any) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def render_html(report: Report, charts: Optional[dict[str, Any]] = None) -> str:
    """Render the report to an HTML string. Charts: {name: plotly_figure}."""
    charts = charts or report.charts
    template = _load_template("report.html.j2")
    return template.render(
        report=report,
        profile=report.profile,
        quality=report.quality,
        automl=report.automl,
        charts=charts,
        chart_json={name: _fig_to_json(fig) for name, fig in charts.items()},
        report_json=report.to_json(),
    )


def export_html(report: Report, path: Union[str, Path]) -> Path:
    """Write the interactive HTML report to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(report)
    p.write_text(html, encoding="utf-8")
    return p
