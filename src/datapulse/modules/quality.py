"""Data quality: duplicates, outliers (IQR / IsolationForest), drift (PSI)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import polars as pl

from datapulse.report import QualityReport


def detect_duplicates(df: pl.DataFrame) -> int:
    """Count exact duplicate rows."""
    return int(df.height - df.unique(keep="first").height)


def outlier_counts_iqr(df: pl.DataFrame, multiplier: float = 1.5) -> Dict[str, int]:
    """Count outliers per numeric column using the IQR rule."""
    counts: Dict[str, int] = {}
    for name, dtype in df.schema.items():
        if dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
            continue
        s = df[name].drop_nulls()
        if s.len() < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr is None or float(iqr) == 0.0:
            continue
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out:
            counts[name] = n_out
    return counts


def outlier_counts_isolation_forest(
    df: pl.DataFrame, contamination: float = 0.05, max_rows: int = 100_000
) -> Dict[str, int]:
    """Outlier counts using sklearn IsolationForest on numeric columns (sampled)."""
    from sklearn.ensemble import IsolationForest

    numeric_cols = [
        name
        for name, dtype in df.schema.items()
        if dtype
        in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
    ]
    if len(numeric_cols) < 2:
        return {}
    sample = df.select(numeric_cols).drop_nulls()
    if sample.height > max_rows:
        sample = sample.sample(n=max_rows, seed=42)
    if sample.height < 50:
        return {}
    X = sample.to_numpy()
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = model.fit_predict(X)
    flagged = sample.filter(pl.Series(preds) == -1)
    counts: Dict[str, int] = {}
    for col in numeric_cols:
        n = int(flagged[col].drop_nulls().len())
        if n:
            counts[col] = n
    return counts


def psi(expected: pl.Series, actual: pl.Series, buckets: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    clean_e = expected.drop_nulls()
    clean_a = actual.drop_nulls()
    if clean_e.len() < 2 or clean_a.len() < 2:
        return 0.0
    lo = min(float(clean_e.min()), float(clean_a.min()))
    hi = max(float(clean_e.max()), float(clean_a.max()))
    if lo == hi:
        return 0.0
    edges = [lo + i * (hi - lo) / buckets for i in range(buckets + 1)]
    edges[-1] = float("inf")

    def dist(s: pl.Series) -> List[float]:
        counts = [0.0] * buckets
        for v in s.to_list():
            for i in range(buckets):
                if v >= edges[i] and v < edges[i + 1]:
                    counts[i] += 1
                    break
        total = sum(counts)
        return [c / total if total else 0.0 for c in counts]

    de, da = dist(clean_e), dist(clean_a)
    eps = 1e-6
    return float(sum((a - e) * math.log((a + eps) / (e + eps)) for e, a in zip(de, da)))


def drift_report(
    reference: pl.DataFrame, current: pl.DataFrame, numeric_buckets: int = 10, threshold: float = 0.2
) -> Dict[str, Dict[str, float]]:
    """PSI drift per shared numeric column. PSI > 0.1 = moderate, > 0.25 = significant."""
    common = [
        name for name in reference.columns if name in current.columns
        and reference[name].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    result: Dict[str, Dict[str, float]] = {}
    for name in common:
        psi_val = psi(reference[name], current[name], buckets=numeric_buckets)
        result[name] = {"psi": round(psi_val, 4), "drift": psi_val > threshold}
    return result


def quality_score(profile, duplicates: int, outlier_cols: Dict[str, int]) -> float:
    """Composite 0..1 quality score from profile, duplicates, and outliers."""
    if profile is None or profile.rows == 0:
        return 0.0
    score = 1.0
    null_cols = [c for c in profile.column_profiles if c.null_pct > 50]
    score -= 0.2 * len(null_cols) / max(1, profile.columns)
    score -= 0.3 * min(1.0, duplicates / max(1, profile.rows))
    if outlier_cols:
        score -= 0.1 * min(1.0, len(outlier_cols) / max(1, profile.columns))
    return round(max(0.0, score), 3)


def analyze_quality(df: pl.DataFrame, profile=None, use_isolation_forest: bool = True) -> QualityReport:
    """Run the full quality pipeline over a DataFrame."""
    dups = detect_duplicates(df)
    if use_isolation_forest and df.height >= 50:
        outlier_cols = outlier_counts_isolation_forest(df)
    else:
        outlier_cols = outlier_counts_iqr(df)
    score = quality_score(profile, dups, outlier_cols)
    return QualityReport(
        duplicate_rows=dups,
        duplicate_pct=round(dups / max(1, df.height) * 100, 4),
        outlier_columns=outlier_cols,
        quality_score=score,
    )
