"""Automatic feature engineering: datetime parsing, text vectorization, encoding, scaling, interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import polars as pl

from datapulse.config import FeatureEngineeringConfig

_NUMERIC = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
_DATETIME = (pl.Datetime, pl.Date, pl.Time, pl.Duration)
_STRING = (pl.String, pl.Categorical)


@dataclass
class FeatureEngineeringResult:
    df: pl.DataFrame
    engineered_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in _NUMERIC


def parse_datetimes(df: pl.DataFrame, max_cols: int = 10) -> tuple[pl.DataFrame, List[str]]:
    """Parse string columns that look like ISO datetimes into Date/Datetime."""
    new_cols: List[str] = []
    string_cols = [name for name, dtype in df.schema.items() if dtype in _STRING][:max_cols]
    for name in string_cols:
        sample = df[name].drop_nulls().head(50).to_list()
        if not sample:
            continue
        looks_date = all(isinstance(v, str) and ("-" in v or "/" in v) and len(v) >= 8 for v in sample)
        if not looks_date:
            continue
        has_time = any((":" in v or "T" in v) for v in sample)
        try:
            parsed = pl.col(name).str.to_datetime(format=None, strict=False)
            if not has_time:
                parsed = parsed.cast(pl.Date)
            df = df.with_columns(parsed.alias(name + "__parsed"))
            new_cols.append(name + "__parsed")
        except Exception:
            continue
    return df, new_cols


def encode_categoricals(
    df: pl.DataFrame, encoding: str = "auto", max_cardinality: int = 50, target: Optional[str] = None
) -> tuple[pl.DataFrame, List[str]]:
    """Ordinal-encode low-cardinality categorical columns; keep high-cardinality as-is."""
    new_cols: List[str] = []
    cat_cols = [name for name, dtype in df.schema.items() if dtype in _STRING]
    for name in cat_cols:
        if name == target:
            continue
        n_unique = df[name].n_unique()
        if n_unique > max_cardinality:
            continue
        mapping = {
            v: i for i, v in enumerate(sorted(df[name].drop_nulls().unique().to_list()))
        }
        if not mapping:
            continue
        df = df.with_columns(pl.col(name).replace(mapping).cast(pl.Float64).alias(name + "__enc"))
        new_cols.append(name + "__enc")
    return df, new_cols


def extract_datetime_features(df: pl.DataFrame) -> tuple[pl.DataFrame, List[str]]:
    """Extract year/month/dayofweek from Date/Datetime columns."""
    new_cols: List[str] = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, (pl.Datetime, pl.Date)):
            df = df.with_columns(
                pl.col(name).dt.year().alias(name + "__year"),
                pl.col(name).dt.month().alias(name + "__month"),
                pl.col(name).dt.weekday().alias(name + "__dow"),
            )
            new_cols += [name + "__year", name + "__month", name + "__dow"]
    return df, new_cols


def scale_numeric(df: pl.DataFrame, method: str = "robust", exclude: List[str] | None = None) -> tuple[pl.DataFrame, List[str]]:
    """Standard/Robust scale numeric columns (sklearn) producing __scaled columns."""
    exclude = exclude or []
    numeric_cols = [name for name, dtype in df.schema.items() if _is_numeric(dtype) and name not in exclude]
    clean = [c for c in numeric_cols if df[c].null_count() == 0]
    if not clean:
        return df, []
    from sklearn.preprocessing import RobustScaler, StandardScaler

    scaler = RobustScaler() if method == "robust" else StandardScaler()
    X = df.select(clean).to_numpy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xs = scaler.fit_transform(X)
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)
    scaled = pl.DataFrame(Xs, schema=clean, orient="row")
    scaled = scaled.rename({c: c + "__scaled" for c in clean})
    new_cols = list(scaled.columns)
    df = df.hstack(scaled)
    return df, new_cols


def add_interactions(df: pl.DataFrame, max_pairs: int = 100, exclude: List[str] | None = None) -> tuple[pl.DataFrame, List[str]]:
    """Add pairwise numeric interactions for the most variable columns."""
    exclude = exclude or []
    numeric_cols = [name for name, dtype in df.schema.items() if _is_numeric(dtype) and name not in exclude]
    clean = [c for c in numeric_cols if df[c].null_count() == 0 and c.endswith("__scaled")]
    if len(clean) < 2:
        return df, []
    cols = clean[: min(len(clean), 30)]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]][:max_pairs]
    new_cols: List[str] = []
    for a, b in pairs:
        name = f"{a}x{b}"
        df = df.with_columns((pl.col(a) * pl.col(b)).alias(name))
        new_cols.append(name)
    return df, new_cols


def mutual_information_importance(df: pl.DataFrame, target: str) -> Dict[str, float]:
    """Mutual information between numeric features and target."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    features = [c for c in df.columns if _is_numeric(df[c].dtype) and c != target and df[c].null_count() == 0]
    if len(features) < 1 or df[target].null_count() > 0:
        return {}
    X = df.select(features).head(50_000).to_numpy()
    y = df[target].head(50_000).to_numpy()
    if df[target].dtype in _STRING:
        y = LabelEncoder().fit_transform(y)
        fn = mutual_info_classif
    else:
        fn = mutual_info_regression
    try:
        mi = fn(X, y, random_state=42)
        return {f: round(float(m), 4) for f, m in zip(features, mi) if m > 0}
    except Exception:
        return {}


def run_feature_engineering(
    df: pl.DataFrame,
    config: Optional[FeatureEngineeringConfig] = None,
    target: Optional[str] = None,
) -> FeatureEngineeringResult:
    """Run the full feature engineering pipeline over a DataFrame."""
    config = config or FeatureEngineeringConfig()
    engineered: List[str] = []
    dropped: List[str] = []
    target_cols = [target] if target else []
    # engineered columns derived from the target would leak it into the model
    leak_prefixes = tuple(f"{target}__" for target in target_cols)

    def _is_leak(col: str) -> bool:
        return col in target_cols or col.startswith(leak_prefixes)

    if config.datetime_parsing:
        df, cols = parse_datetimes(df)
        engineered += cols
        df, cols = extract_datetime_features(df)
        engineered += cols

    if config.encoding != "none":
        df, cols = encode_categoricals(df, encoding=config.encoding, target=target)
        engineered += cols

    if config.scaling != "none":
        df, cols = scale_numeric(df, method=config.scaling, exclude=target_cols)
        engineered += [c for c in cols if not _is_leak(c)]

    if config.interactions:
        df, cols = add_interactions(df, exclude=target_cols)
        engineered += [c for c in cols if not _is_leak(c)]

    # drop engineered columns derived from the target to prevent leakage
    leak_cols = [c for c in df.columns if _is_leak(c) and c not in target_cols]
    if leak_cols:
        df = df.drop(leak_cols)
        dropped += leak_cols

    # cap total columns to avoid explosion
    cap = config.max_features
    if len(df.columns) > cap:
        keep = [c for c in df.columns if c in (target or [])] or []
        df = df.select(df.columns[: cap - len(keep)] + keep)
        dropped = df.columns[: max(0, len(df.columns) - cap)]

    return FeatureEngineeringResult(df=df, engineered_columns=engineered, dropped_columns=dropped)
