"""Plotly chart builders for the analysis report."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import polars as pl

from datapulse.report import AutoMLResult, DatasetProfile, QualityReport

_NUMERIC = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(font_size=12),
)


def _fig(data, **layout) -> go.Figure:
    layout = {**_LAYOUT, **layout}
    return go.Figure(data=data, layout=layout)


def dtype_pie(df: pl.DataFrame) -> go.Figure:
    counts: Dict[str, int] = {}
    for dtype in df.schema.values():
        key = str(dtype).split("[")[0]
        counts[key] = counts.get(key, 0) + 1
    return _fig(
        go.Pie(labels=list(counts.keys()), values=list(counts.values()), hole=0.45),
        title="Column type distribution",
    )


def missingness_bar(profile: DatasetProfile) -> go.Figure:
    cols = [c.name for c in profile.column_profiles if c.null_pct > 0]
    pcts = [c.null_pct for c in profile.column_profiles if c.null_pct > 0]
    if not cols:
        cols = ["(no missing values)"]
        pcts = [0]
    return _fig(
        go.Bar(x=cols, y=pcts, marker_color="#f59e0b"),
        title="Missing values by column (%)",
        yaxis_title="null %",
    )


def histogram(df: pl.DataFrame, col: str, max_points: int = 50_000) -> Optional[go.Figure]:
    if df[col].dtype not in _NUMERIC:
        return None
    s = df[col].drop_nulls()
    if s.len() > max_points:
        s = s.sample(n=max_points, seed=42)
    return _fig(
        go.Histogram(x=s.to_list(), nbinsx=40, marker_color="#2563eb"),
        title=f"Distribution of {col}",
        xaxis_title=col,
        yaxis_title="count",
    )


def scatter(df: pl.DataFrame, x: str, y: str, color: Optional[str] = None, max_points: int = 20_000) -> Optional[go.Figure]:
    if df[x].dtype not in _NUMERIC or df[y].dtype not in _NUMERIC:
        return None
    s = df.drop_nulls(subset=[x, y])
    if s.height > max_points:
        s = s.sample(n=max_points, seed=42)
    data: List[Any] = []
    if color and color in s.columns and s[color].dtype not in _NUMERIC:
        for cat in s[color].drop_nulls().unique().to_list()[:12]:
            sub = s.filter(pl.col(color) == cat)
            data.append(
                go.Scattergl(
                    x=sub[x].to_list(),
                    y=sub[y].to_list(),
                    mode="markers",
                    name=str(cat),
                    marker=dict(size=4, opacity=0.6),
                )
            )
    else:
        data.append(
            go.Scattergl(
                x=s[x].to_list(),
                y=s[y].to_list(),
                mode="markers",
                marker=dict(size=4, opacity=0.6, color="#2563eb"),
            )
        )
    return _fig(data, title=f"{x} vs {y}", xaxis_title=x, yaxis_title=y)


def correlation_heatmap(corr: Dict[str, Dict[str, float]]) -> Optional[go.Figure]:
    cols = list(corr.keys())
    if len(cols) < 2:
        return None
    z = [[corr[a][b] for b in cols] for a in cols]
    return _fig(
        go.Heatmap(
            z=z,
            x=cols,
            y=cols,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="r"),
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        ),
        title="Pearson correlation matrix",
        height=600,
    )


def box_cat_num(df: pl.DataFrame, cat: str, num: str, max_points: int = 20_000) -> Optional[go.Figure]:
    if df[num].dtype not in _NUMERIC or df[cat].dtype in _NUMERIC:
        return None
    s = df.drop_nulls(subset=[cat, num])
    if s.height > max_points:
        s = s.sample(n=max_points, seed=42)
    data = [
        go.Box(
            y=s.filter(pl.col(cat) == c)[num].to_list(),
            name=str(c),
            boxmean="sd",
        )
        for c in s[cat].drop_nulls().unique().to_list()[:12]
    ]
    return _fig(data, title=f"{num} by {cat}", yaxis_title=num, xaxis_title=cat)


def category_bar(df: pl.DataFrame, col: str, max_cats: int = 15) -> Optional[go.Figure]:
    if df[col].dtype in _NUMERIC:
        return None
    vc = df[col].drop_nulls().value_counts().sort("count", descending=True).head(max_cats)
    return _fig(
        go.Bar(x=vc[col].to_list(), y=vc["count"].to_list(), marker_color="#10b981"),
        title=f"Top values in {col}",
        xaxis_title=col,
        yaxis_title="count",
    )


def leaderboard_bar(automl: AutoMLResult) -> go.Figure:
    names = [m.name for m in automl.leaderboard]
    means = [m.cv_mean for m in automl.leaderboard]
    colors = ["#2563eb" if m.name == automl.best_model_name else "#94a3b8" for m in automl.leaderboard]
    return _fig(
        go.Bar(x=names, y=means, marker_color=colors, error_y=dict(type="data", array=[m.cv_std for m in automl.leaderboard])),
        title=f"AutoML leaderboard ({automl.metric})",
        yaxis_title=automl.metric,
    )


def importance_bar(importance: Dict[str, float], title: str = "Feature importance", top: int = 20) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top]
    names, vals = zip(*items) if items else ([], [])
    return _fig(
        go.Bar(x=list(vals), y=list(names), orientation="h", marker_color="#7c3aed"),
        title=title,
        xaxis_title="importance",
        height=min(120 + 24 * len(names), 900),
    )


def duplicate_quality_bars(quality: QualityReport, profile: DatasetProfile) -> go.Figure:
    labels = ["duplicates", "quality score"]
    values = [quality.duplicate_pct, quality.quality_score * 100]
    return _fig(
        go.Bar(x=labels, y=values, marker_color=["#ef4444", "#22c55e"]),
        title="Duplicates % and quality score",
        yaxis_title="%",
    )


def build_charts(
    df: pl.DataFrame,
    profile: DatasetProfile,
    quality: QualityReport,
    automl: Optional[AutoMLResult] = None,
    explainability: Any = None,
    max_points: int = 20_000,
) -> Dict[str, go.Figure]:
    """Build the standard set of charts for a report."""
    charts: Dict[str, go.Figure] = {}
    charts["dtype_pie"] = dtype_pie(df)

    missing = missingness_bar(profile)
    if missing is not None:
        charts["missingness"] = missing

    numeric_cols = [c.name for c in profile.column_profiles if c.is_numeric]
    cat_cols = [c.name for c in profile.column_profiles if not c.is_numeric and c.cardinality <= 50]

    # top 8 numeric histograms
    for col in numeric_cols[:8]:
        h = histogram(df, col, max_points)
        if h is not None:
            charts[f"hist_{col}"] = h

    # a few scatter pairs (top numeric by importance if automl, else first two)
    if len(numeric_cols) >= 2:
        color = cat_cols[0] if cat_cols else None
        s = scatter(df, numeric_cols[0], numeric_cols[1], color=color, max_points=max_points)
        if s is not None:
            charts["scatter_main"] = s

    # category bars
    for col in cat_cols[:4]:
        b = category_bar(df, col)
        if b is not None:
            charts[f"cat_{col}"] = b

    # box plots cat x num
    if cat_cols and numeric_cols:
        b = box_cat_num(df, cat_cols[0], numeric_cols[0], max_points=max_points)
        if b is not None:
            charts["box_main"] = b

    # correlation heatmap
    corr = profile.correlations.get("pearson", {})
    if not corr and len(numeric_cols) >= 2:
        from datapulse.modules.profiling import correlation_matrix

        corr = correlation_matrix(df, "pearson")
        profile.correlations["pearson"] = corr
    hm = correlation_heatmap(corr)
    if hm is not None:
        charts["corr_heatmap"] = hm

    charts["quality"] = duplicate_quality_bars(quality, profile)

    if automl is not None:
        charts["leaderboard"] = leaderboard_bar(automl)
        if automl.feature_importance:
            charts["fe_importance"] = importance_bar(automl.feature_importance, "Model feature importance")

    if explainability is not None and explainability.global_importance:
        charts["shap_importance"] = importance_bar(
            explainability.global_importance, f"SHAP global importance ({explainability.method})"
        )

    return charts
