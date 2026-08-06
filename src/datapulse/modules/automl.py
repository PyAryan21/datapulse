"""AutoML: task detection, cross-validated model leaderboard, Optuna tuning.

Models: LightGBM, XGBoost, CatBoost, RandomForest, Linear (classification/regression).
Uses sklearn pipelines with median imputation; capped row count for speed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datapulse.config import AutoMLConfig
from datapulse.report import AutoMLResult, ModelScore

_NUMERIC = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in _NUMERIC


def detect_task(y: pl.Series) -> str:
    """Infer classification vs regression from target dtype/cardinality."""
    if y.dtype in (pl.String, pl.Categorical, pl.Boolean):
        return "classification"
    n_unique = y.drop_nulls().n_unique()
    if n_unique <= 20:
        return "classification"
    return "regression"


def select_metric(task: str) -> Tuple[str, Any]:
    if task == "classification":
        return "f1_weighted", make_scorer(f1_score, average="weighted", zero_division=0)
    return "rmse", "neg_root_mean_squared_error"


def _prepare(df: pl.DataFrame, target: str, task: str, max_rows: int = 200_000) -> Tuple[Any, Any, List[str]]:
    """Build X (numpy), y (numpy) from numeric features; cap rows.

    Drops features with near-perfect correlation to the target (leakage guard).
    """
    features = [c for c in df.columns if c != target and _is_numeric(df[c].dtype)]
    # keep only finite features; drop all-null ones
    features = [c for c in features if df[c].drop_nulls().len() > 0]
    data = df.select(features + [target]).drop_nulls().drop_nans()
    if data.height > max_rows:
        data = data.sample(n=max_rows, seed=42)
    if data.height < 50:
        raise ValueError(f"Not enough clean rows ({data.height}) for AutoML; need >= 50")

    # leakage guard: drop features with |corr| >= 0.95 with the target
    dropped_leaks: List[str] = []
    if data[target].dtype in _NUMERIC and len(features) > 0:
        clean_features = [c for c in features if data[c].null_count() == 0]
        for c in clean_features:
            try:
                r = pl.select(pl.corr(pl.lit(data[c]), pl.lit(data[target]))).item()
                if r is not None and abs(float(r)) >= 0.95:
                    dropped_leaks.append(c)
            except Exception:
                continue
        if dropped_leaks:
            features = [c for c in features if c not in dropped_leaks]

    X = data.select(features).to_numpy()
    y_series = data[target]
    if task == "classification" and y_series.dtype not in (pl.String, pl.Categorical, pl.Boolean):
        # ordinal encode to ints
        y = y_series.cast(pl.Float64).round().cast(pl.Int64).to_numpy()
    else:
        y = y_series.to_numpy()
    return X, y, features


_MODEL_FACTORIES: Dict[str, Any] = {
    "LightGBM": lambda: __import__("lightgbm").LGBMClassifier(n_estimators=200, verbose=-1, random_state=42),
    "XGBoost": lambda: __import__("xgboost").XGBClassifier(n_estimators=200, verbosity=0, random_state=42),
    "CatBoost": lambda: __import__("catboost").CatBoostClassifier(iterations=200, verbose=0, random_state=42),
    "RandomForest": lambda: __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "Linear": lambda: LogisticRegression(max_iter=2000, n_jobs=-1),
}
_MODEL_FACTORIES_REGRESSION: Dict[str, Any] = {
    "LightGBM": lambda: __import__("lightgbm").LGBMRegressor(n_estimators=200, verbose=-1, random_state=42),
    "XGBoost": lambda: __import__("xgboost").XGBRegressor(n_estimators=200, verbosity=0, random_state=42),
    "CatBoost": lambda: __import__("catboost").CatBoostRegressor(iterations=200, verbose=0, random_state=42),
    "RandomForest": lambda: __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "Linear": lambda: LinearRegression(),
}

_IMPUTER_PIPE = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)


def _build_pipeline(model: Any) -> Pipeline:
    return Pipeline([("prep", _IMPUTER_PIPE), ("model", model)])


def _cv_scheme(task: str, y: Any, folds: int):
    if task == "classification" and len(set(y.tolist() if hasattr(y, "tolist") else y)) > 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def train_leaderboard(
    df: pl.DataFrame,
    target: str,
    config: Optional[AutoMLConfig] = None,
    progress_cb=None,
) -> AutoMLResult:
    """Run AutoML: CV leaderboard over configured models + train best model."""
    config = config or AutoMLConfig()
    task = config.task.value if config.task.value != "auto" else detect_task(df[target])
    if config.task.value != "auto" and task != config.task.value:
        task = config.task.value
    metric_name, scorer = select_metric(task)
    X, y, features = _prepare(df, target, task, max_rows=config.max_rows)

    factories = _MODEL_FACTORIES_REGRESSION if task == "regression" else _MODEL_FACTORIES
    models_to_run = [m for m in config.models if m in factories]
    if not models_to_run:
        models_to_run = list(factories.keys())

    leaderboard: List[ModelScore] = []
    best_model = None
    best_score = float("-inf")
    cv_scheme = _cv_scheme(task, y, config.cv_folds)
    start = time.monotonic()

    for i, name in enumerate(models_to_run):
        if progress_cb:
            progress_cb(name, i + 1, len(models_to_run))
        t0 = time.monotonic()
        try:
            pipe = _build_pipeline(factories[name]())
            scores = cross_val_score(pipe, X, y, cv=cv_scheme, scoring=scorer, n_jobs=1)
            mean_s, std_s = float(scores.mean()), float(scores.std())
            # negative scorer (rmse) -> invert for leaderboard ordering
            if metric_name == "rmse":
                mean_s, std_s = -mean_s, std_s
            leaderboard.append(ModelScore(name=name, cv_mean=round(mean_s, 4), cv_std=round(std_s, 4), metric=metric_name))
            if mean_s > best_score:
                best_score = mean_s
                pipe.fit(X, y)
                best_model = pipe
        except Exception as e:
            leaderboard.append(ModelScore(name=name, cv_mean=0.0, cv_std=0.0, metric=metric_name, params={"error": str(e)}))

    leaderboard.sort(key=lambda m: m.cv_mean, reverse=True)

    result = AutoMLResult(
        task=task,
        target=target,
        metric=metric_name,
        leaderboard=leaderboard,
        best_model_name=leaderboard[0].name if leaderboard else None,
        train_time_s=round(time.monotonic() - start, 2),
    )
    # feature importance from best model
    if best_model is not None:
        try:
            imp = _feature_importance(best_model, features, X, y)
            result.feature_importance = imp
        except Exception:
            pass
    result._best_pipeline = best_model  # type: ignore[attr-defined]
    result._X = X  # type: ignore[attr-defined]
    result._y = y  # type: ignore[attr-defined]
    result._features = features  # type: ignore[attr-defined]
    return result


def _feature_importance(pipe: Pipeline, features: List[str], X: Any, y: Any) -> Dict[str, float]:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if imp is not None and len(imp) == len(features):
            return {f: round(float(v), 5) for f, v in zip(features, imp) if v > 0}
    # linear fallback: |coef| standardized by prep scaler
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return {f: round(abs(float(c)), 5) for f, c in zip(features, coef)}
    return {}


def tune_lightgbm(
    df: pl.DataFrame,
    target: str,
    trials: int = 30,
    task: Optional[str] = None,
    max_rows: int = 50_000,
) -> Dict[str, Any]:
    """Optuna hyperparameter tuning for LightGBM (returns best params)."""
    import optuna
    import lightgbm as lgb

    task = task or detect_task(df[target])
    X, y, features = _prepare(df, target, task, max_rows=max_rows)
    cv_scheme = _cv_scheme(task, y, 3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
            "random_state": 42,
        }
        cls = lgb.LGBMClassifier(**params) if task == "classification" else lgb.LGBMRegressor(**params)
        scores = cross_val_score(_build_pipeline(cls), X, y, cv=cv_scheme, n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params
