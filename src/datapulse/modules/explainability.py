"""Explainability: SHAP global importance + permutation importance on the best model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from datapulse.report import AutoMLResult, ExplainabilityResult


def explain_best_model(
    automl: AutoMLResult,
    max_samples: int = 2000,
    method: str = "auto",
) -> Optional[ExplainabilityResult]:
    """Compute SHAP global importance for the best model in an AutoMLResult."""
    pipe = automl._best_pipeline
    X = automl._X
    features = automl._features
    if pipe is None or X is None or features is None:
        return None

    model = pipe.named_steps["model"]
    model_name = type(model).__name__.lower()
    y_full = automl._y

    # subsample for speed
    idx = None
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        Xs = X[idx]
        ys = y_full[idx] if y_full is not None else None
    else:
        Xs = X
        ys = y_full

    shap = _import_shap()
    try:
        if method == "auto":
            if any(k in model_name for k in ("lgbm", "xgb", "catboost", "randomforest", "extra")):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xs)
            else:
                masker = shap.maskers.Independent(Xs, max_samples=500)
                explainer = shap.PermutationExplainer(pipe.predict, masker)
                shap_values = explainer.shap_values(Xs)

        if isinstance(shap_values, list):
            # multiclass: aggregate over classes
            shap_values = np.abs(np.stack(shap_values)).mean(axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = np.abs(shap_values).mean(axis=0)

        global_imp = {
            f: float(np.abs(shap_values[:, i]).mean()) for i, f in enumerate(features)
        }
        base_value = None
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            base_value = float(np.mean(ev) if isinstance(ev, (list, np.ndarray)) else ev)

        global_imp = dict(sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True))
        return ExplainabilityResult(
            method=type(explainer).__name__,
            global_importance=global_imp,
            sample_used=len(Xs),
            base_value=base_value,
        )
    except Exception:
        # fallback: permutation importance
        try:
            from sklearn.inspection import permutation_importance

            pi = permutation_importance(pipe, Xs, ys, n_repeats=5, random_state=42)
            if pi.importances_mean is None:
                return None
            global_imp = {f: float(pi.importances_mean[i]) for i, f in enumerate(features)}
            return ExplainabilityResult(
                method="permutation_importance",
                global_importance=dict(sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True)),
                sample_used=len(Xs),
            )
        except Exception:
            return None


def _import_shap():
    import shap  # noqa: F401

    return shap
