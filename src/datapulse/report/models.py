"""Core report data models (Pydantic)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    cardinality: int
    is_unique: bool = False
    is_numeric: bool = False
    is_datetime: bool = False
    stats: Dict[str, Optional[float]] = Field(default_factory=dict)
    datetime_range: Optional[Tuple[str, str]] = None
    top_values: List[Tuple[Any, int]] = Field(default_factory=list)
    histogram: List[Tuple[float, float, int]] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    memory_mb: float
    column_profiles: List[ColumnProfile] = Field(default_factory=list)
    correlations: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
    missing_summary: Dict[str, float] = Field(default_factory=dict)


class QualityReport(BaseModel):
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    outlier_columns: Dict[str, int] = Field(default_factory=dict)
    quality_score: float = 1.0


class FeatureEngineeringReport(BaseModel):
    original_columns: int = 0
    engineered_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    total_columns: int = 0


class ModelScore(BaseModel):
    name: str
    cv_mean: float
    cv_std: float
    metric: str
    params: Dict[str, Any] = Field(default_factory=dict)


class AutoMLResult(BaseModel):
    task: Literal["classification", "regression"]
    target: str
    metric: str
    leaderboard: List[ModelScore] = Field(default_factory=list)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    cv_scores: List[float] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    train_time_s: Optional[float] = None


class ExplainabilityResult(BaseModel):
    method: str = ""
    global_importance: Dict[str, float] = Field(default_factory=dict)
    sample_used: int = 0
    base_value: Optional[float] = None


class TimeSeriesResult(BaseModel):
    is_timeseries: bool = False
    time_column: Optional[str] = None
    trend: Optional[Dict[str, float]] = None
    seasonality_period: Optional[int] = None
    seasonality_strength: Optional[float] = None


class TextResult(BaseModel):
    columns: List[str] = Field(default_factory=list)
    stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    datapulse_version: str = ""
    source: str = ""
    duration_s: float = 0.0


class Report(BaseModel):
    """Complete analysis report — serializable to JSON, rendered to HTML/MD."""

    metadata: RunMetadata = RunMetadata()
    profile: Optional[DatasetProfile] = None
    quality: QualityReport = QualityReport()
    feature_engineering: Optional[FeatureEngineeringReport] = None
    automl: Optional[AutoMLResult] = None
    explainability: Optional[ExplainabilityResult] = None
    timeseries: Optional[TimeSeriesResult] = None
    text: Optional[TextResult] = None
    charts: Dict[str, Any] = Field(default_factory=dict, description="Serialized Plotly figures")

    # --- exporters -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        s = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(s, encoding="utf-8")
        return s if path is None else None
