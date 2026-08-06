"""Configuration system: Pydantic models + YAML config file loading."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class Task(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AutoMLConfig(BaseModel):
    task: Task = Task.AUTO
    target: Optional[str] = None
    time_budget: int = Field(default=1800, ge=30)
    models: List[str] = Field(
        default_factory=lambda: ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "Linear"]
    )
    cv_folds: int = Field(default=5, ge=2, le=10)
    metric: str = "auto"
    optuna_trials: int = Field(default=50, ge=1, le=1000)
    max_rows: int = Field(default=200_000, description="Cap rows fed to AutoML for speed")


class FeatureEngineeringConfig(BaseModel):
    enabled: bool = True
    datetime_parsing: bool = True
    text_vectorization: Literal["tfidf", "embeddings", "none"] = "tfidf"
    encoding: Literal["auto", "target", "onehot", "ordinal"] = "auto"
    scaling: Literal["standard", "minmax", "robust", "none"] = "robust"
    interactions: bool = True
    selection: Literal["mi", "none"] = "mi"
    max_features: int = Field(default=200, ge=10, le=5000)


class VisualizationConfig(BaseModel):
    theme: Literal["auto", "light", "dark"] = "auto"
    palette: str = "plotly"
    sampling: int = Field(default=50_000, description="Max points per chart")
    cross_filter: bool = True


class ModuleConfig(BaseModel):
    profiling: bool = True
    quality: bool = True
    feature_engineering: bool = True
    automl: bool = True
    explainability: bool = True
    timeseries: bool = False
    text: bool = False
    unstructured: bool = False


class InputConfig(BaseModel):
    sql: Optional[str] = None
    sample_rows: Optional[int] = None
    stratify_column: Optional[str] = None
    chunk_size: int = Field(default=100_000, ge=1000)
    sheet_name: Union[int, str] = 0


class HtmlConfig(BaseModel):
    standalone: bool = True
    embed_assets: bool = True


class NotebookConfig(BaseModel):
    include_code: bool = True
    execute: bool = False


class OutputConfig(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["html", "json"])
    html: HtmlConfig = HtmlConfig()
    notebook: NotebookConfig = NotebookConfig()


class AnalysisConfig(BaseModel):
    """Top-level analysis configuration."""

    input: InputConfig = InputConfig()
    modules: ModuleConfig = ModuleConfig()
    automl: AutoMLConfig = AutoMLConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def validate_target_required(self) -> "AnalysisConfig":
        if self.modules.automl and self.automl.target is None:
            # target may be provided later via CLI/API; not an error here
            pass
        return self

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AnalysisConfig":
        """Load config from a YAML file, merging onto defaults."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(**raw)

    def apply_overrides(self, overrides: Dict[str, Any]) -> "AnalysisConfig":
        """Apply CLI/API overrides on top of config (nested keys with dot notation)."""
        data = self.model_dump()
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return AnalysisConfig(**data)

    def dump_yaml(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(exclude_none=True, mode="json"), sort_keys=False),
            encoding="utf-8",
        )


def config_from_env() -> AnalysisConfig:
    """Create config, honoring DATAPULSE_* environment variable overrides.

    Mapping: DATAPULSE_MODULES_AUTOML=0, DATAPULSE_AUTOML_TIME_BUDGET=600, ...
    """
    base = AnalysisConfig()
    overrides: Dict[str, Any] = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("DATAPULSE_"):
            continue
        key = env_key[len("DATAPULSE_") :].lower().replace("_", ".")
        if not key:
            continue
        try:
            overrides[key] = yaml.safe_load(env_val)
        except yaml.YAMLError:
            overrides[key] = env_val
    return base.apply_overrides(overrides)
