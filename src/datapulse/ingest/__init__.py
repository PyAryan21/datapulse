"""Universal data ingestion: auto-detect format and load into Polars LazyFrame."""

from datapulse.ingest.readers import (
    detect_format,
    load_dataframe,
    load_lazyframe,
    load_sql,
    sample_for_viz,
)

__all__ = ["detect_format", "load_dataframe", "load_lazyframe", "load_sql", "sample_for_viz"]
