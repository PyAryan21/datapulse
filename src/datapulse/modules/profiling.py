"""Data profiling: schema, stats, distributions, correlations, missingness."""

from __future__ import annotations

from typing import List

import polars as pl

from datapulse.report import ColumnProfile, DatasetProfile


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def _is_datetime(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.Datetime, pl.Date, pl.Time, pl.Duration))


def _histogram(series: pl.Series, bins: int = 40) -> List[tuple[float, float, int]]:
    clean = series.drop_nulls()
    vals = [v for v in clean.to_list() if v == v]
    if len(vals) < 2:
        return []
    lo = float(min(vals))
    hi = float(max(vals))
    if lo == hi:
        return [(lo, hi, len(vals))]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, c) for i, c in enumerate(counts)]


def profile_dataset(df: pl.DataFrame) -> DatasetProfile:
    """Profile an eager Polars DataFrame into a DatasetProfile."""
    n_rows, n_cols = df.shape
    profiles: List[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        null_count = int(series.null_count())
        n = max(1, n_rows)
        null_pct = null_count / n * 100.0
        cardinality = int(series.n_unique())
        numeric = _is_numeric(dtype)
        datetime_ = _is_datetime(dtype)

        stats: dict[str, float | None] = {}
        if numeric:
            clean = series.drop_nulls()
            if len(clean):
                qs = clean.quantile([0.25, 0.5, 0.75])
                stats = {
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                    "q1": float(qs[0]),
                    "median": float(qs[1]),
                    "q3": float(qs[2]),
                    "skew": float(clean.skew() if len(clean) > 2 else 0.0),
                    "kurtosis": float(clean.kurtosis() if len(clean) > 3 else 0.0),
                }
        elif datetime_:
            clean = series.drop_nulls()
            if len(clean):
                stats = {}
                datetime_range = (str(clean.min()), str(clean.max()))
            else:
                datetime_range = None

        top_values = []
        if not numeric and not datetime_ and n > 0:
            vc = series.value_counts()[:10]
            top_values = [(row[0], int(row[1])) for row in vc.iter_rows()]

        hist = _histogram(series) if numeric else []

        profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(dtype),
                null_count=null_count,
                null_pct=round(null_pct, 4),
                cardinality=cardinality,
                is_unique=(cardinality == n_rows and n_rows > 1),
                is_numeric=numeric,
                is_datetime=datetime_,
                stats=stats,
                datetime_range=datetime_range if datetime_ else None,
                top_values=top_values,
                histogram=hist,
            )
        )

    missing_summary = {c.name: round(c.null_pct, 2) for c in profiles if c.null_count > 0}
    memory_mb = df.estimated_size("mb")

    return DatasetProfile(
        rows=n_rows,
        columns=n_cols,
        memory_mb=round(memory_mb, 3),
        column_profiles=profiles,
        missing_summary=missing_summary,
    )


def correlation_matrix(df: pl.DataFrame, method: str = "pearson") -> dict[str, dict[str, float]]:
    """Pearson/Spearman correlation matrix over numeric columns."""
    numeric_cols = [name for name, dtype in df.schema.items() if _is_numeric(dtype)]
    if len(numeric_cols) < 2:
        return {}
    out: dict[str, dict[str, float]] = {c: {} for c in numeric_cols}
    try:
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i:]:
                val = pl.select(pl.corr(pl.lit(df[col_a]), pl.lit(df[col_b]), method=method)).item()
                if val is None:
                    val = float("nan")
                out[col_a][col_b] = round(float(val), 4)
                out[col_b][col_a] = round(float(val), 4)
        return out
    except Exception:
        return {}
