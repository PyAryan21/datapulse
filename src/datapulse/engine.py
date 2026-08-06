"""Analysis orchestrator: runs the configured module pipeline over a source."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from datapulse import __version__
from datapulse.config import AnalysisConfig
from datapulse.ingest import load_lazyframe, sample_for_viz
from datapulse.modules.feature_engineering import run_feature_engineering
from datapulse.modules.profiling import correlation_matrix, profile_dataset
from datapulse.modules.quality import analyze_quality
from datapulse.report import (
    DatasetProfile,
    FeatureEngineeringReport,
    Report,
    RunMetadata,
)


def run_analysis(
    source: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
    sql: Optional[str] = None,
) -> Report:
    """Run the full analysis pipeline over a data source.

    ``source`` can be a file path, directory, or a SQLAlchemy URI.
    """
    config = config or AnalysisConfig()
    if target:
        config = config.apply_overrides({"automl.target": target})
    if task:
        config = config.apply_overrides({"automl.task": task})
    if sql:
        config = config.apply_overrides({"input.sql": sql})

    start = time.monotonic()
    report = Report(metadata=RunMetadata(datapulse_version=__version__, source=str(source)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.completed} of {task.total}"),
        transient=True,
    ) as progress:
        step = progress.add_task("Loading data", total=1)
        lf = _load(source, config)
        df = lf.collect()
        progress.update(step, completed=1)

        step = progress.add_task("Profiling", total=1)
        report.profile = profile_dataset(df)
        corr = correlation_matrix(df, "pearson")
        spearman = correlation_matrix(df, "spearman")
        report.profile.correlations = {
            "pearson": corr,
            "spearman": spearman,
        }
        progress.update(step, completed=1)

        if config.modules.quality:
            step = progress.add_task("Quality checks", total=1)
            report.quality = analyze_quality(df, profile=report.profile)
            progress.update(step, completed=1)

        if config.modules.feature_engineering:
            step = progress.add_task("Feature engineering", total=1)
            fe = run_feature_engineering(df, config.feature_engineering, target=config.automl.target)
            report.feature_engineering = FeatureEngineeringReport(
                original_columns=len(df.columns),
                engineered_columns=fe.engineered_columns,
                dropped_columns=fe.dropped_columns,
                total_columns=len(fe.df.columns),
            )
            report._engineered_df = fe.df

        if config.modules.automl:
            if config.automl.target is None:
                report.warnings.append(
                    "AutoML and explainability skipped: no target column provided. "
                    "Pass --target <column> to enable model training."
                )
            else:
                step = progress.add_task("AutoML", total=1)
                from datapulse.modules.automl import train_leaderboard

                df_for_automl = report._engineered_df if report._engineered_df is not None else df
                report.automl = train_leaderboard(
                    df_for_automl,
                    config.automl.target,
                    config=config.automl,
                    progress_cb=lambda name, i, n: progress.update(
                        step, description=f"AutoML [{i}/{n}] {name}"
                    ),
                )
                progress.update(step, completed=1)

                if config.modules.explainability:
                    step = progress.add_task("Explainability", total=1)
                    from datapulse.modules.explainability import explain_best_model

                    report.explainability = explain_best_model(report.automl)
                    progress.update(step, completed=1)

    report.metadata.duration_s = round(time.monotonic() - start, 3)
    return report


def _load(source: Union[str, Path], config: AnalysisConfig) -> pl.LazyFrame:
    path = Path(source)
    sql = config.input.sql
    if path.exists():
        return load_lazyframe(path, sql=sql, sheet_name=config.input.sheet_name)
    # assume SQLAlchemy URI
    from datapulse.ingest.readers import load_sql

    if not sql:
        raise ValueError(
            f"Source {source!r} is not a file/directory; provide --sql for database sources"
        )
    return load_sql(str(source), sql)


def summarize(report: Report) -> Dict[str, Any]:
    """Human-readable summary dict of a report."""
    profile = report.profile or DatasetProfile(rows=0, columns=0, memory_mb=0.0)
    return {
        "rows": profile.rows,
        "columns": profile.columns,
        "memory_mb": profile.memory_mb,
        "missing_columns": len(profile.missing_summary),
        "duplicate_rows": report.quality.duplicate_rows,
        "quality_score": report.quality.quality_score,
        "best_model": report.automl.best_model_name if report.automl else None,
        "duration_s": report.metadata.duration_s,
    }
