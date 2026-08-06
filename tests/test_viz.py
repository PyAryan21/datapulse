import numpy as np
import polars as pl

from datapulse.modules.quality import analyze_quality
from datapulse.modules.profiling import profile_dataset
from datapulse.viz.charts import build_charts, correlation_heatmap, scatter


def _sample_df():
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "d": rng.choice(["2023-01-01", "2023-06-01", "2023-12-31"], n),
        }
    )


def test_build_charts_contains_core():
    df = _sample_df()
    profile = profile_dataset(df)
    quality = analyze_quality(df, profile=profile)
    charts = build_charts(df, profile, quality)
    for key in ["dtype_pie", "corr_heatmap", "scatter_main", "cat_cat", "box_main", "quality"]:
        assert key in charts, f"missing chart {key}"


def test_scatter_none_on_non_numeric():
    df = _sample_df()
    assert scatter(df, "cat", "d") is None
    assert scatter(df, "x", "y") is not None


def test_correlation_heatmap_small():
    df = _sample_df()
    corr = {"x": {"x": 1.0, "y": 0.1}, "y": {"x": 0.1, "y": 1.0}}
    fig = correlation_heatmap(corr)
    assert fig is not None
    assert len(fig.data) == 1
