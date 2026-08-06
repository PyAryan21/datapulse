"""Format detectors and readers.

All readers return a Polars ``LazyFrame`` for streaming/out-of-core processing.
CSV/TSV/JSON reading uses Polars' native streaming engine; SQL uses SQLAlchemy
connect strings with chunked fetch.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import polars as pl

_FORMAT_EXTENSIONS: Dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".arrow": "feather",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xlsx": "excel",
    ".xls": "excel",
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".avro": "avro",
    ".orc": "orc",
}

_FORMAT_MAGIC: List[tuple[str, str]] = [
    (b"PAR1", "parquet"),
    (b"{\"", "json"),
    (b"[{", "json"),
]


def detect_format(path: Union[str, Path]) -> str:
    """Detect format from file extension, falling back to magic bytes."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _FORMAT_EXTENSIONS:
        return _FORMAT_EXTENSIONS[ext]
    # magic bytes fallback for extension-less files
    try:
        head = p.read_bytes()[:4]
        for magic, fmt in _FORMAT_MAGIC:
            if head.startswith(magic):
                return fmt
    except OSError:
        pass
    return "csv"


def load_lazyframe(
    source: Union[str, Path],
    format: Optional[str] = None,
    sql: Optional[str] = None,
    sheet_name: Union[int, str] = 0,
    chunk_size: int = 100_000,
    **kwargs: Any,
) -> pl.LazyFrame:
    """Load any supported source into a Polars LazyFrame (streaming)."""
    path = Path(source)
    fmt = format or detect_format(path)

    if fmt == "sqlite":
        return _read_sqlite(path, sql)
    if sql:
        raise ValueError("--sql is only supported with SQLite sources for now")

    if fmt == "csv":
        return pl.scan_csv(path, infer_schema_length=10_000, **kwargs)
    if fmt == "tsv":
        return pl.scan_csv(path, separator="\t", infer_schema_length=10_000, **kwargs)
    if fmt == "parquet":
        return pl.scan_parquet(path, **kwargs)
    if fmt == "feather":
        return pl.scan_ipc(path, **kwargs)
    if fmt == "json":
        return pl.scan_ndjson(path, **kwargs) if path.suffix == ".jsonl" else _read_json(path)
    if fmt == "jsonl":
        return pl.scan_ndjson(path, **kwargs)
    if fmt == "excel":
        df = pl.read_excel(path, sheet_name=sheet_name)
        return df.lazy()
    if fmt == "avro":
        return pl.scan_avro(path, **kwargs)
    if fmt == "orc":
        df = pl.read_orc(path)
        return df.lazy()
    raise ValueError(f"Unsupported format: {fmt!r}")


def _read_json(path: Path) -> pl.LazyFrame:
    """JSON may be a single object or array of objects — sniff and load."""
    head = path.read_text(encoding="utf-8", errors="replace")[:512].lstrip()
    if head.startswith("["):
        df = pl.read_json(path)
    else:
        df = pl.read_ndjson(path)
    return df.lazy()


def _read_sqlite(path: Path, sql: Optional[str]) -> pl.LazyFrame:
    if not sql:
        raise ValueError("SQLite sources require --sql 'SELECT ...'")
    conn = sqlite3.connect(path)
    try:
        df = pl.read_database(sql, conn)
    finally:
        conn.close()
    return df.lazy()


def load_sql(connection_uri: str, query: str) -> pl.LazyFrame:
    """Load from any SQLAlchemy-compatible database (postgres, mysql, sqlite)."""
    from sqlalchemy import create_engine

    engine = create_engine(connection_uri)
    df = pl.read_database(query, engine.connect())
    engine.dispose()
    return df.lazy()


def load_dataframe(
    source: Union[str, Path],
    **kwargs: Any,
) -> pl.DataFrame:
    """Load source eagerly into a Polars DataFrame."""
    return load_lazyframe(source, **kwargs).collect()


def sample_for_viz(lf: pl.LazyFrame, n: int, stratify: Optional[str] = None) -> pl.DataFrame:
    """Sample a LazyFrame for visualization, optionally stratified."""
    if stratify is not None:
        counts = lf.select(pl.col(stratify)).collect().to_series()
        per_class = max(1, n // max(1, len(counts.unique())))
        return (
            lf.group_by(stratify)
            .agg(pl.all().sample(n=per_class, with_replacement=False))
            .explode(pl.all().exclude(stratify))
            .collect()
        )
    return lf.collect().sample(n=min(n, lf.select(pl.len()).collect().item()), seed=42)
