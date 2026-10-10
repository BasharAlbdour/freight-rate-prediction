"""Model definitions. All models predict log(posted_rate)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from . import features as F

FINAL_MODEL_NAME = "6. Final: calendar-aware linear + boosted residuals"
LEVEL_WINDOW_DAYS = 14

DEFAULT_BOOST_PARAMS = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=63,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42,
)


@dataclass
class LinearPlusBoost:
    """Log-linear structural model plus gradient boosting on its residuals.

    * The linear part captures the dominant, smooth structure: rate grows with
      distance, equipment premiums, market level, and the calendar terms (linear
      trend and quarter-end ramp). Being linear it extrapolates safely into the
      unseen Nov-Dec period, which trees cannot do.
    * The boosted part learns what is left: geography (via coordinates),
      non-linear distance effects, interactions and the date-level market signals.

    With ``linear_cols=[]`` it degenerates to a pure boosted model (centred on the
    mean), and with ``boost_cols=[]`` to a pure linear model.
    """

    linear_cols: list[str] = field(default_factory=list)
    boost_cols: list[str] = field(default_factory=list)
    boost_params: dict = field(default_factory=lambda: dict(DEFAULT_BOOST_PARAMS))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearPlusBoost":
        y = np.asarray(y, dtype=float)
        if self.linear_cols:
            self.linear_ = LinearRegression().fit(X[self.linear_cols], y)
            base = self.linear_.predict(X[self.linear_cols])
        else:
            self.linear_ = None
            self.mean_ = float(y.mean())
            base = np.full(len(y), self.mean_)
        if self.boost_cols:
            self.boost_ = HistGradientBoostingRegressor(**self.boost_params).fit(X[self.boost_cols], y - base)
        else:
            self.boost_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.linear_ is not None:
            pred = self.linear_.predict(X[self.linear_cols])
        else:
            pred = np.full(len(X), self.mean_)
        if self.boost_ is not None:
            pred = pred + self.boost_.predict(X[self.boost_cols])
        return pred


@dataclass
class RecentLevelAdjusted:
    """Optional wrapper: add the model's recent mean log-residual to every prediction.

    Tested in validation (reports/level_carryover.csv) as a persistence forecast of
    any remaining market-state drift. Once the calendar terms are in the model the
    gain is small (~1%), so the final model does not use it.
    """

    model: object
    window_days: int = LEVEL_WINDOW_DAYS

    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: pd.Series) -> "RecentLevelAdjusted":
        y = np.asarray(y, dtype=float)
        self.model.fit(X, y)
        recent = (dates > dates.max() - pd.Timedelta(days=self.window_days)).to_numpy()
        self.level_ = float(np.mean(y[recent] - self.model.predict(X[recent])))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X) + self.level_


def build_final(
    boost_params: dict | None = None,
    boost_cols: list[str] | None = None,
    linear_cols: list[str] | None = None,
) -> LinearPlusBoost:
    """The chosen model: calendar-aware log-linear structure + boosted residuals."""
    return LinearPlusBoost(
        list(F.LINEAR_FINAL if linear_cols is None else linear_cols),
        list(F.BOOST_ALL if boost_cols is None else boost_cols),
        dict(boost_params or DEFAULT_BOOST_PARAMS),
    )


def candidate_models(boost_params: dict | None = None) -> dict[str, Callable[[], LinearPlusBoost]]:
    """Factories for the models compared in validation, from naive baseline to the final model.

    Factories (not instances) so every fold fits a fresh, unfitted model.
    """
    params = dict(boost_params or DEFAULT_BOOST_PARAMS)
    return {
        "1. Baseline: log-linear (distance, equipment)": lambda: LinearPlusBoost(F.LINEAR_BASE, []),
        "2. Linear (all features + calendar terms)": lambda: LinearPlusBoost(F.LINEAR_ALL, []),
        "3. Gradient boosting only": lambda: LinearPlusBoost([], F.BOOST_ALL, params),
        "4. Linear + boosted residuals (no calendar terms)": lambda: LinearPlusBoost(F.LINEAR_STRUCTURE, F.BOOST_ALL, params),
        "5. As 4 + linear trend only (no quarter-end ramp)": lambda: LinearPlusBoost(
            F.LINEAR_STRUCTURE + ["t_days"], F.BOOST_ALL, params
        ),
        FINAL_MODEL_NAME: lambda: build_final(params),
    }
