"""Validation schemes and metrics.

Why not a random split?  The labels carry a smooth, strongly autocorrelated
daily "market state" (lag-1 autocorrelation of the daily residual ~0.96). A random
split places neighbouring days in both train and test, so a model can look good
by interpolating that state. The real task predicts the *future* (Nov-Dec 2025)
from the past (Jan-Oct 2025), so the primary scheme is forward-chaining with a
two-month test horizon, the same horizon as the real validation set. A random
5-fold CV is also computed, purely to show how optimistic it would be.

A second concern is that eight cities appear only in the validation file. The
unseen-city scheme trains without a set of cities and tests on loads that touch
them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .data import TARGET
from .models import RecentLevelAdjusted

# (name, test start, test end) - the training window is everything before test start.
FORWARD_FOLDS = [
    ("May-Jun", "2025-05-01", "2025-07-01"),
    ("Jul-Aug", "2025-07-01", "2025-09-01"),
    ("Sep-Oct", "2025-09-01", "2025-11-01"),
]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs(y_pred - y_true) / y_true) * 100),
        "R2": r2_score(y_true, y_pred),
    }


def _score(frame: pd.DataFrame, pred_log: np.ndarray, scheme: str, fold: str, model: str) -> list[dict]:
    """Score on all rows ("raw", labels as delivered) and on rows without corrupted labels ("clean")."""
    rows = []
    pred = np.exp(pred_log)
    for subset, mask in (("raw", np.ones(len(frame), bool)), ("clean", ~frame["label_outlier"].to_numpy())):
        m = regression_metrics(frame[TARGET].to_numpy()[mask], pred[mask])
        rows.append({"model": model, "scheme": scheme, "fold": fold, "subset": subset, "n_test": int(mask.sum()), **m})
    return rows


def _fit_predict(
    make_model: Callable,
    data: pd.DataFrame,
    X: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    level_window_days: int | None = None,
) -> np.ndarray:
    """Fit on clean training rows, predict log-rate for the test rows.

    ``level_window_days`` optionally wraps the model in ``RecentLevelAdjusted``
    (carry the mean residual of the last N training days forward).
    """
    fit_mask = train_mask & ~data["label_outlier"].to_numpy()
    y = np.log(data[TARGET].to_numpy())
    model = make_model()
    if level_window_days is None:
        model.fit(X[fit_mask], y[fit_mask])
    else:
        model = RecentLevelAdjusted(model, window_days=level_window_days)
        model.fit(X[fit_mask], y[fit_mask], data.loc[fit_mask, "date"])
    return model.predict(X[test_mask])


def forward_chaining(
    models: dict[str, Callable],
    data: pd.DataFrame,
    X: pd.DataFrame,
    level_window_days: int | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    rows = []
    for name, make_model in models.items():
        for fold, start, end in FORWARD_FOLDS:
            train_mask = (data["date"] < start).to_numpy()
            test_mask = ((data["date"] >= start) & (data["date"] < end)).to_numpy()
            pred = _fit_predict(make_model, data, X, train_mask, test_mask, level_window_days)
            rows += _score(data[test_mask], pred, label or "forward-chaining", fold, name)
    return pd.DataFrame(rows)


def random_kfold(models: dict[str, Callable], data: pd.DataFrame, X: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Shuffled K-fold. Shown for contrast only - it does NOT mimic the real task."""
    rows = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, make_model in models.items():
        for k, (tr, te) in enumerate(splitter.split(data), start=1):
            train_mask = np.zeros(len(data), bool)
            test_mask = np.zeros(len(data), bool)
            train_mask[tr], test_mask[te] = True, True
            pred = _fit_predict(make_model, data, X, train_mask, test_mask)
            rows += _score(data[test_mask], pred, "random 5-fold (optimistic)", f"fold{k}", name)
    return pd.DataFrame(rows)


def unseen_cities(
    models: dict[str, Callable],
    data: pd.DataFrame,
    X: pd.DataFrame,
    n_groups: int = 4,
    group_size: int = 8,
    fold: tuple[str, str, str] = FORWARD_FOLDS[-1],
) -> pd.DataFrame:
    """Forward fold where ``group_size`` cities are completely hidden from training.

    Test rows are the loads in the test window touching a hidden city (as origin or
    destination), mirroring the 8 cities that only occur in validation.csv.
    """
    cities = np.array(sorted(set(data["pickup"]) | set(data["delivery"])))
    rng = np.random.default_rng(7)
    _, start, end = fold
    in_time_train = (data["date"] < start).to_numpy()
    in_time_test = ((data["date"] >= start) & (data["date"] < end)).to_numpy()
    rows = []
    for g in range(n_groups):
        hidden = set(rng.choice(cities, size=group_size, replace=False))
        touches = (data["pickup"].isin(hidden) | data["delivery"].isin(hidden)).to_numpy()
        train_mask = in_time_train & ~touches
        test_mask = in_time_test & touches
        for name, make_model in models.items():
            pred = _fit_predict(make_model, data, X, train_mask, test_mask)
            rows += _score(data[test_mask], pred, "unseen cities (Sep-Oct)", f"group{g + 1}", name)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, subset: str = "clean") -> pd.DataFrame:
    """Average the per-fold metrics for one subset, one row per (scheme, model)."""
    sub = results[results["subset"] == subset]
    return (
        sub.groupby(["scheme", "model"], sort=False)[["MAE", "RMSE", "MAPE_%", "R2"]]
        .mean()
        .round({"MAE": 1, "RMSE": 1, "MAPE_%": 3, "R2": 5})
        .reset_index()
    )
