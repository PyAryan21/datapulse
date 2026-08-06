# Datapulse

> Any data. Any format. Complete ML-grade analysis. Instantly.

Datapulse is a local-first, Python CLI + library that ingests datasets of **any format**,
runs a **complete automated ML analysis** (profiling, quality, feature engineering, AutoML,
explainability, time series, NLP, unstructured data), and exports **interactive multi-format
reports** (HTML, JSON, Notebook, Markdown, PDF) with publication-quality visualizations.

## Features

- **Universal ingestion** — CSV, TSV, Parquet, Feather, Excel, JSON/JSONL, SQL, Delta Lake, Iceberg, Hudi, Avro, ORC, PDF
- **Full ML pipeline** — data profiling, quality checks, auto feature engineering, AutoML leaderboards, SHAP explainability
- **Interactive visualizations** — scatter, heatmaps, pairplots, violin, beeswarm, waterfall, and more (Plotly)
- **Multi-format export** — self-contained interactive HTML, JSON summary, reproducible Jupyter notebook, Markdown
- **Scale-ready** — Polars streaming + DuckDB handle datasets larger than RAM
- **Zero-config** — `datapulse analyze file.csv` just works

## Installation

```bash
pip install -e ".[all]"
```

## Quick Start

```bash
# Full analysis, zero config
datapulse analyze data.csv

# With target column and task
datapulse analyze train.parquet --target price --task regression

# Multi-format export
datapulse analyze data.csv --format html,json,ipynb,md --output-dir ./reports

# Config-driven
datapulse analyze data/ --config datapulse.yaml
```

### Python API

```python
import datapulse

report = datapulse.analyze("data.csv", target="Survived", task="classification")
report.to_html("report.html")
report.to_json("report.json")
```

## Roadmap

See [PRD.md](PRD.md) for the full product requirements document.

- [x] Phase 1: Foundation (ingest, config, CLI, profiling, HTML export)
- [ ] Phase 2: Quality + feature engineering
- [ ] Phase 3: AutoML + explainability
- [ ] Phase 4: Specialized modules (time series, text, unstructured)
- [ ] Phase 5: Export polish (PDF, notebooks, themes)
- [ ] Phase 6: Beta hardening

## License

MIT
