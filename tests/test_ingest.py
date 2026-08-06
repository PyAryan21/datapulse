import polars as pl
import pytest

from datapulse.ingest import detect_format, load_dataframe, load_lazyframe
from datapulse.modules.profiling import correlation_matrix, profile_dataset


@pytest.fixture(scope="module")
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "b": [10, 20, 30, 40, 50, 60],
            "cat": ["x", "y", "x", "z", "y", "x"],
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), interval="1mo", eager=True),
        }
    )


def test_detect_format(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"")
    assert detect_format(p) == "parquet"
    assert detect_format(tmp_path / "x.csv") == "csv"
    assert detect_format(tmp_path / "x.jsonl") == "jsonl"
    assert detect_format(tmp_path / "x.sqlite3") == "sqlite"


def test_roundtrip_csv(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.write_csv(p)
    loaded = load_dataframe(p)
    assert loaded.shape == sample_df.shape
    assert loaded.columns == sample_df.columns
    lf = load_lazyframe(p)
    assert lf.collect().shape == sample_df.shape


def test_roundtrip_parquet(tmp_path, sample_df):
    p = tmp_path / "data.parquet"
    sample_df.write_parquet(p)
    loaded = load_dataframe(p)
    assert loaded.shape == sample_df.shape


def test_profile_dataset(sample_df):
    prof = profile_dataset(sample_df)
    assert prof.rows == 6
    assert prof.columns == 4
    a = next(c for c in prof.column_profiles if c.name == "a")
    assert a.null_count == 1
    assert a.null_pct == pytest.approx(16.6667, abs=0.01)
    assert a.is_numeric
    assert a.stats["min"] == 1.0
    assert a.stats["mean"] == pytest.approx(3.0)
    assert a.histogram, "numeric column should have a histogram"
    cat = next(c for c in prof.column_profiles if c.name == "cat")
    assert not cat.is_numeric
    assert len(cat.top_values) > 0


def test_correlation_matrix(sample_df):
    corr = correlation_matrix(sample_df, "pearson")
    assert set(corr.keys()) == {"a", "b"}
    assert corr["a"]["b"] == pytest.approx(1.0, abs=1e-4)


def test_profile_handles_nan(tmp_path):
    df = pl.DataFrame({"x": [float("nan"), 1.0, 2.0, None, 3.0]})
    prof = profile_dataset(df)
    x = next(c for c in prof.column_profiles if c.name == "x")
    assert x.stats["min"] == 1.0
    assert x.stats["max"] == 3.0
