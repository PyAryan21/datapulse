# Product Requirements Document (PRD)
## Datapulse — Universal Dataset Analyzer
**Tagline:** *Any data. Any format. Complete ML-grade analysis. Instantly.*

---

### 1. Executive Summary

**Product Name:** Datapulse (codename: UDA — Universal Dataset Analyzer)

**Vision:** A local-first, Python-based CLI + library that ingests datasets of any format (tabular, big data, unstructured), runs comprehensive automated ML analysis (EDA + feature engineering + AutoML + explainability), and exports interactive multi-format reports (HTML, PDF, Jupyter Notebook, JSON, Markdown) with publication-quality visualizations.

**Target Users:** Data scientists, ML engineers, analysts, researchers who need rapid, thorough dataset understanding without writing boilerplate.

**Core Value Prop:** Zero-config, runs locally on medium-to-large data (100MB–10GB+), supports everything from CSV to Delta Lake to PDFs, outputs everything you'd manually do in a week — in minutes.

---

### 2. Problem Statement

Current pain points:
- **Format fragmentation:** Different tools for CSV, Parquet, SQL, Delta, PDFs, images
- **Analysis fragmentation:** Separate tools for profiling, AutoML, explainability, feature engineering
- **Scale gaps:** pandas breaks >RAM; Spark/Dask too heavy for local exploration
- **Report gaps:** Static HTML or manual notebooks; no multi-format export
- **Config overhead:** Every tool needs tuning; no "just work" mode

**Datapulse solves this:** One command, any input, complete analysis, beautiful interactive report, multiple export formats.

---

### 3. Scope

#### In Scope (v1.0 - MVP)
- **Input Formats:** CSV, TSV, Parquet, Feather, Excel, JSON/JSONL, SQL (SQLite, PostgreSQL, MySQL), Delta Lake, Iceberg, Hudi, Avro, ORC, PDF (tabular extraction), Images (basic metadata), Audio/Video (metadata)
- **Core Engine:** Polars (streaming/out-of-core) + DuckDB (SQL analytics) + Arrow (interop)
- **Analysis Modules:**
  - Data Profiling (schema, stats, distributions, missingness, correlations, cardinality)
  - Data Quality (duplicates, outliers, drift detection, schema validation)
  - Feature Engineering (auto: datetime parsing, text vectorization, encoding, scaling, interactions, embeddings)
  - AutoML (classification/regression: model selection, CV, hyperparameter tuning, leaderboards)
  - Explainability (SHAP global/local, permutation importance, partial dependence, counterfactuals)
  - Time Series (decomposition, seasonality, forecasting baselines)
  - Text/NLP (token stats, embeddings, topic modeling, sentiment)
  - Image/Audio/Video (metadata, embeddings via CLIP/Whisper if available)
- **Visualizations:** Interactive Plotly (scatter, heatmap, bar, violin, box, histogram, pairplot, parallel coords, dendrogram, treemap, sankey, 3D, geo)
- **Output Formats:** Interactive HTML (single file), PDF, Jupyter Notebook (.ipynb), JSON summary, Markdown
- **Interface:** CLI (`datapulse analyze ...`), Python API (`datapulse.analyze(df)`), optional local web UI
- **Config:** YAML config for module toggles, thresholds, model selection, viz themes

#### Out of Scope (v1.0)
- Distributed training (Ray/Spark cluster mode) — local only
- Real-time streaming analytics
- Model deployment/serving
- Collaboration/versioning
- Cloud hosting/SaaS

#### Future (v1.1+)
- Plugin system for custom formats/analyzers
- Distributed mode (Dask/Ray)
- LLM-assisted insights (natural language summaries)
- Active learning loop
- Data lineage tracking

---

### 4. Functional Requirements

#### FR-1: Universal Data Ingestion
| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Auto-detect format from extension/content (magic bytes) | P0 |
| FR-1.2 | Stream large files (>RAM) via Polars streaming + DuckDB | P0 |
| FR-1.3 | SQL connector: read from any DB via SQLAlchemy + chunked fetch | P0 |
| FR-1.4 | Delta Lake / Iceberg / Hudi: read latest snapshot via Delta-rs | P0 |
| FR-1.5 | PDF: extract tables (pdfplumber) + text (PyMuPDF) | P1 |
| FR-1.6 | Images: EXIF + CLIP embeddings (optional, lazy import) | P1 |
| FR-1.7 | Audio/Video: metadata + Whisper transcription (optional) | P2 |
| FR-1.8 | Unified internal representation: Polars DataFrame + Arrow schema | P0 |

#### FR-2: Automated Analysis Pipeline
| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | **Profiling:** Schema inference, dtypes, null %, cardinality, min/max/mean/std/quantiles, histograms, correlations (Pearson/Spearman/MI), target leakage detection | P0 |
| FR-2.2 | **Quality:** Duplicate rows (exact), outlier detection (IQR, IsolationForest), data drift (PSI, KS-test vs reference), schema validation | P0 |
| FR-2.3 | **Feature Engineering:** Auto datetime parsing, text TF-IDF, categorical encoding, scaling, polynomial interactions, feature selection (mutual info, SHAP) | P0 |
| FR-2.4 | **AutoML:** Classification/Regression — LightGBM, XGBoost, CatBoost, RandomForest, Linear with CV, Optuna tuning, leaderboard | P0 |
| FR-2.5 | **Explainability:** SHAP (TreeExplainer, Permutation), global feature importance, dependence plots, force plots, waterfall | P0 |
| FR-2.6 | **Time Series:** STL decomposition, ACF/PACF, seasonality detection, Prophet/NeuralForecast baselines | P1 |
| FR-2.7 | **Text:** Token stats, vocab size, embeddings (sentence-transformers), BERTopic, sentiment | P1 |
| FR-2.8 | **Unstructured:** CLIP image embeddings, Whisper audio transcripts, frame sampling for video | P2 |

#### FR-3: Visualization Engine
| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Plotly-based interactive charts (zoom, hover, select, export PNG/SVG) | P0 |
| FR-3.2 | Chart types: Scatter (2D/3D), Heatmap (corr, MI, SHAP), Bar/Histogram, Violin/Box, Pairplot, Parallel Coordinates, Dendrogram, Treemap, Sankey, Geo, Time Series, Partial Dependence, SHAP beeswarm/waterfall | P0 |
| FR-3.3 | Smart sampling for large data (stratified, reservoir) to keep charts responsive | P0 |
| FR-3.4 | Theme system: light/dark, colorblind-safe palettes, publication style | P1 |
| FR-3.5 | Cross-filtering: click bar → highlight scatter points | P1 |

#### FR-4: Multi-Format Report Export
| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | **Interactive HTML:** Single self-contained file (Plotly.js embedded), collapsible sections, search, TOC, download charts | P0 |
| FR-4.2 | **PDF:** Playwright render of HTML → paginated PDF | P0 |
| FR-4.3 | **Jupyter Notebook:** .ipynb with markdown narrative, code cells (reproducible), outputs | P0 |
| FR-4.4 | **JSON:** Machine-readable summary (metrics, model scores, feature importance, config) | P0 |
| FR-4.5 | **Markdown:** GitHub-ready summary with embedded chart images | P1 |

#### FR-5: CLI & Python API
| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | `datapulse analyze <input> -o <output> --format html,json` | P0 |
| FR-5.2 | `datapulse analyze <input> --target <col> --task classification` | P0 |
| FR-5.3 | `datapulse analyze <input> --config datapulse.yaml` | P0 |
| FR-5.4 | Python: `report = datapulse.analyze(df, target="y", task="regression")` | P0 |
| FR-5.5 | Python: `report.to_html()`, `report.to_json()` etc. | P0 |
| FR-5.6 | Progress bars (rich/tqdm), structured logging, error recovery | P0 |

#### FR-6: Configuration & Extensibility
| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | YAML config: enable/disable modules, set thresholds, model list, viz theme | P0 |
| FR-6.2 | Environment variable overrides | P1 |
| FR-6.3 | Plugin hooks (future): custom analyzers, format readers, chart types | P2 |

---

### 5. Non-Functional Requirements

| Category | Requirement | Target |
|---|---|---|
| **Performance** | Stream 10GB Parquet on 16GB RAM | < 5 min full analysis |
| **Performance** | AutoML on 1M rows × 100 cols | < 30 min (configurable budget) |
| **Memory** | Peak RSS ≤ 1.5× dataset size (streaming) | |
| **Startup** | Cold CLI start → first output | < 3 sec |
| **Reliability** | Graceful degradation: skip optional deps if not installed | |
| **Portability** | Linux, macOS, Windows (Python 3.10+) | |
| **Security** | No network calls unless explicitly enabled (model download) | |
| **Licensing** | MIT, dependency licenses compatible | |

---

### 6. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      datapulse CLI / API                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Ingest    │  │  Config    │  │  Plugins   │
    │  Layer     │  │  Manager   │  │  (future)  │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          ▼               ▼               ▼
    ┌────────────────────────────────────────────────┐
    │              Analysis Orchestrator              │
    │  (DAG execution, parallelization, caching)      │
    └────────────────────────┬───────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Profiling    │    │  Feature Eng  │    │   AutoML      │
│  + Quality    │    │  (auto)       │    │  + Explain    │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                    ┌───────────────┐
                    │  Report       │
                    │  Builder      │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   HTML   │  │   JSON   │  │ Notebook │
        └──────────┘  └──────────┘  └──────────┘
```

**Key Components:**
- **Ingest Layer:** Format detectors → Polars LazyFrame (streaming)
- **Orchestrator:** Topological sort of analysis modules, disk cache (joblib)
- **Modules:** Independent, stateless, configurable, return standardized `AnalysisResult` (Pydantic)
- **Report Builder:** Jinja2 templates + Plotly.js + custom JS for interactivity
- **Exporters:** HTML (template), JSON (Pydantic dump), Notebook (nbformat), Markdown, PDF (Playwright)

---

### 7. CLI Design

```bash
# Basic usage
datapulse analyze data.csv --output report.html

# With target + task
datapulse analyze train.parquet --target price --task regression --output report/

# Multi-format
datapulse analyze data.db --sql "SELECT * FROM events" --format html,json,ipynb,md --output-dir ./reports

# Config file
datapulse analyze data/ --config datapulse.yaml

# Sampling for speed
datapulse analyze huge.parquet --sample 100000 --stratify target

# Module control
datapulse analyze data.csv --skip automl,explainability

# Theme
datapulse analyze data.csv --theme dark --palette viridis
```

---

### 8. Implementation Phases

#### Phase 1: Foundation
- Project scaffolding (pyproject.toml, src/datapulse/, tests/)
- Core ingest: Polars readers for CSV, Parquet, JSON, SQL, Excel
- Config system (Pydantic Settings + YAML)
- CLI framework (Typer + Rich)
- Basic profiling module (schema, stats, histograms, correlations)
- HTML report template (Jinja2 + Plotly.js)
- Unit tests for ingest + profiling

#### Phase 2: Quality + Feature Engineering
- Quality module: duplicates, outliers (IQR, IsolationForest), drift (PSI)
- Feature engineering: datetime, text TF-IDF, encoding, scaling, interactions
- Sampling utilities for large data visualization
- Extend HTML report with quality + FE sections

#### Phase 3: AutoML + Explainability
- AutoML integration (Optuna + LightGBM/XGB/CatBoost/RF)
- Model leaderboard, CV, prediction artifacts
- SHAP integration (TreeExplainer, Permutation)
- Explainability charts: beeswarm, waterfall, dependence, force
- Notebook exporter (nbformat)

#### Phase 4: Specialized Modules
- Time series module (STL, ACF/PACF, Prophet baseline)
- Text module (token stats, sentence-transformers, BERTopic)
- Unstructured: PDF tables (pdfplumber), images (CLIP), audio (Whisper)
- Delta Lake / Iceberg / Hudi readers

#### Phase 5: Export + Polish
- PDF exporter (Playwright), JSON exporter, Markdown exporter
- Themes (light/dark/colorblind), palette system
- Cross-filtering JS in HTML report
- Performance optimization, memory profiling
- Documentation, examples, CI/CD

#### Phase 6: Beta Hardening
- Stress test on 10GB+ datasets
- Edge cases: all-null columns, single-row, high-cardinality, mixed types
- Optional dependency handling (graceful degradation)
- Windows/macOS/Linux CI
- Release v1.0

---

### 9. Dependencies

**Core (always installed):**
```
polars[pyarrow]>=1.0
duckdb>=1.0
plotly>=5.20
scikit-learn>=1.4
lightgbm>=4.3
xgboost>=2.0
catboost>=1.2
shap>=0.44
optuna>=4.0
pydantic>=2.6
pydantic-settings>=2.3
typer>=0.9
rich>=13.7
jinja2>=3.1
nbformat>=5.9
pdfplumber>=0.11
pymupdf>=1.23
pyyaml>=6.0
joblib>=1.4
```

**Optional (lazy import, feature-gated):**
```
delta-rs[polars]          # Delta Lake
pyiceberg[polars]         # Iceberg
sentence-transformers     # Text embeddings
bertopic                  # Topic modeling
openai-whisper            # Audio transcription
prophet                   # Time series forecasting
playwright                # PDF export
```

---

### 10. Success Metrics (v1.0)

- **Usability:** `datapulse analyze file.csv` works zero-config on 10+ formats
- **Speed:** 1M rows × 50 cols full analysis < 10 min on 16GB RAM laptop
- **Completeness:** Report covers profiling, quality, FE, AutoML, explainability
- **Export:** HTML (interactive), PDF (printable), Notebook (reproducible), JSON (programmatic), Markdown (shareable)

---

*End of PRD v1.0*
