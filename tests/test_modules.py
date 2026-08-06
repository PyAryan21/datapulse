import polars as pl

from datapulse.modules.feature_engineering import (
    encode_categoricals,
    extract_datetime_features,
    parse_datetimes,
    run_feature_engineering,
    scale_numeric,
)
from datapulse.modules.quality import (
    detect_duplicates,
    outlier_counts_iqr,
    psi,
)


def test_detect_duplicates():
    df = pl.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    assert detect_duplicates(df) == 1
    assert detect_duplicates(df.unique()) == 0


def test_outlier_iqr():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    counts = outlier_counts_iqr(df)
    assert counts.get("x", 0) == 1


def test_psi():
    a = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert psi(a, a) == 0.0
    shifted = pl.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    assert psi(a, shifted) > 1.0


def test_parse_datetimes():
    df = pl.DataFrame({"d": ["2023-01-01", "2023-06-15", "2024-12-31"]})
    out, cols = parse_datetimes(df)
    assert "d__parsed" in cols
    assert out["d__parsed"].dtype == pl.Date


def test_extract_datetime_features():
    df = pl.DataFrame({"d": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 3), interval="1d", eager=True)})
    out, cols = extract_datetime_features(df)
    assert "d__year" in cols and "d__month" in cols and "d__dow" in cols
    assert out["d__year"].to_list() == [2023, 2023, 2023]


def test_encode_categoricals():
    df = pl.DataFrame({"cat": ["b", "a", "b", "c"]})
    out, cols = encode_categoricals(df)
    assert "cat__enc" in cols
    assert out["cat__enc"].to_list() == [1, 0, 1, 2]


def test_scale_numeric():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, cols = scale_numeric(df, method="standard")
    assert "x__scaled" in cols
    assert abs(float(out["x__scaled"].mean())) < 1e-9
    assert abs(float(out["x__scaled"].std(ddof=0)) - 1.0) < 1e-6


def test_run_feature_engineering_pipeline():
    df = pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "d": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
    )
    result = run_feature_engineering(df)
    assert len(result.engineered_columns) >= 3
    assert "cat__enc" in result.engineered_columns
