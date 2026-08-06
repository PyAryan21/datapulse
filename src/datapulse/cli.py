"""CLI entrypoint: datapulse analyze <input> [options]."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from datapulse.config import AnalysisConfig, config_from_env
from datapulse.engine import run_analysis, summarize

app = typer.Typer(add_completion=False, help="Any data. Any format. Complete ML-grade analysis. Instantly.")
console = Console()


@app.command()
def analyze(
    source: str = typer.Argument(..., help="Path to dataset (file or dir), or DB connection string"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file or directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Comma-separated export formats: html,json,ipynb,md,pdf"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column for AutoML"),
    task: Optional[str] = typer.Option(None, "--task", help="classification | regression | auto"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    sql: Optional[str] = typer.Option(None, "--sql", help="SQL query (for database sources)"),
    sample: Optional[int] = typer.Option(None, "--sample", help="Max rows to analyze"),
    theme: Optional[str] = typer.Option(None, "--theme", help="light | dark | auto"),
    skip: Optional[str] = typer.Option(None, "--skip", help="Comma-separated modules to skip"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run full analysis over a dataset and export reports."""
    cfg = AnalysisConfig.from_yaml(config) if config else config_from_env()

    overrides: dict = {}
    if sample:
        overrides["input.sample_rows"] = sample
    if theme:
        overrides["visualization.theme"] = theme
    if skip:
        for mod in skip.split(","):
            overrides[f"modules.{mod}"] = False
    if format:
        overrides["output.formats"] = [f.strip() for f in format.split(",")]
    cfg = cfg.apply_overrides(overrides)

    console.print(f"[bold cyan]datapulse[/] analyzing [bold]{source}[/]")
    try:
        report = run_analysis(source, cfg, target=target, task=task, sql=sql)
    except ValueError as e:
        console.print(f"[bold red]error: {e}[/]")
        raise typer.Exit(1)

    summary = summarize(report)
    table = Table(title="Analysis Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)
    for w in report.warnings:
        console.print(f"[yellow]warning: {w}[/]")

    # export
    from datapulse.exporters.html_exporter import export_html
    from datapulse.exporters.json_exporter import export_json

    formats = cfg.output.formats
    if output is None:
        out_dir = Path("datapulse_report")
        out_dir.mkdir(exist_ok=True)
        out_base = out_dir / "report"
    elif output.suffix:
        out_base = output
    else:
        out_dir = output
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / "report"

    for fmt in formats:
        fmt = fmt.lower()
        if fmt == "html":
            path = out_base if out_base.suffix == ".html" else Path(str(out_base) + ".html")
            export_html(report, path)
        elif fmt == "json":
            path = out_base if out_base.suffix == ".json" else Path(str(out_base) + ".json")
            export_json(report, path)
        elif fmt in ("ipynb", "md", "pdf"):
            console.print(f"[yellow]âš  {fmt} export not implemented yet (Phase 5)[/]")
        else:
            console.print(f"[yellow]âš  unknown format: {fmt}[/]")

    console.print(f"[green]done in {report.metadata.duration_s}s[/]")


@app.command()
def version():
    """Show version."""
    from datapulse import __version__

    console.print(f"datapulse v{__version__}")


if __name__ == "__main__":
    app()


