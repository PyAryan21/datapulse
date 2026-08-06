import numpy as np
import polars as pl
import pytest

from datapulse.config import AutoMLConfig
from datapulse.modules.automl import detect_task, select_metric, train_leaderboard
from datapulse.modules.explainability import explain_best_model
from datapulse.modules.feature_engineering import run_feature_engineering


@pytest.fixture(scope="module")
def reg_df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 * x1 - 1.5 * x2 + 0.5 * rng.normal(0, 0.5, n)
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": rng.normal(0, 1, n), "cat": rng.choice(["a", "b", "c"], n), "y": y})


@pytest.fixture(scope="module")
def cls_df() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    score = 2 * x1 - x2 + rng.normal(0, 0.3, n)
    y = (score > score.mean()).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": rng.normal(0, 1, n), "y": y})


def test_detect_task(reg_df, cls_df):
    assert detect_task(reg_df["y"]) == "regression"
    assert detect_task(cls_df["y"]) == "classification"
    assert detect_task(pl.Series(["a", "b", "a", "c"])) == "classification"


def test_select_metric():
    name, scorer = select_metric("regression")
    assert name == "rmse"
    name, scorer = select_metric("classification")
    assert name == "f1_weighted"


def test_train_leaderboard_regression(reg_df):
    fe = run_feature_engineering(reg_df, target="y")
    res = train_leaderboard(fe.df, "y")
    assert res.task == "regression"
    assert len(res.leaderboard) >= 3
    # Linear must produce a real score
    linear = next(m for m in res.leaderboard if m.name == "Linear")
    assert linear.cv_mean > 0
    # leaderboard sorted desc by metric (rmse inverted)
    means = [m.cv_mean for m in res.leaderboard]
    assert means == sorted(means, reverse=True)
    assert res.best_model_name is not None
    assert len(res.feature_importance) > 0


def test_train_leaderboard_classification(cls_df):
    res = train_leaderboard(cls_df, "y")
    assert res.task == "classification"
    assert res.metric == "f1_weighted"
    assert res.best_model_name is not None


def test_explainability(reg_df):
    fe = run_feature_engineering(reg_df, target="y")
    res = train_leaderboard(fe.df, "y", config=AutoMLConfig(models=["XGBoost"]))
    expl = explain_best_model(res)
    assert expl is not None
    assert expl.sample_used > 0
    assert len(expl.global_importance) > 0
    names = list(expl.global_importance.keys())
    assert names == sorted(names, key=lambda k: expl.global_importance[k], reverse=True)


def test_leakage_guard():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.1, n)
    leaky = y * 1.0001  # near-perfect copy of target
    df = pl.DataFrame({"x": x, "leaky": leaky, "y": y})
    fe = run_feature_engineering(df, target="y")
    res = train_leaderboard(fe.df, "y", config=AutoMLConfig(models=["Linear"]))
    assert "leaky" not in res.feature_importance
