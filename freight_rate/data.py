"""Loading and cleaning of the freight datasets.

Data-quality findings that drive this module (see reports/eda_summary.json):

* ``weight`` has sign-flipped values (negative), a few missing values and a hard
  cap at 47,500 lb.
* ``market_index`` has ~0.8% missing values. It is a date-level signal plus small
  row-level noise, so a missing value is best recovered from the same date.
* ``posted_rate`` contains ~1.4% corrupted labels that sit at roughly 0.17-0.45x or
  2.2-5.4x the value implied by the load's features. They are removed from
  *training* only.
* Latitude/longitude are synthetic but consistent per city; ``distance`` is
  consistent per lane, so neither is "repaired".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = "posted_rate"
WEIGHT_CAP = 47_500.0
OUTLIER_SIGMA = 5.0


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the labeled development data and the 12,000 loads to predict."""
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    return train, valid


def fit_weight_fill(train: pd.DataFrame) -> pd.Series:
    """Median absolute weight per equipment type (learned on development data only)."""
    return train["weight"].abs().groupby(train["equipment"]).median()


def clean_inputs(frame: pd.DataFrame, weight_fill: pd.Series) -> pd.DataFrame:
    """Repair the load-level input columns. Never touches the target."""
    out = frame.copy()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_capped"] = (out["weight"] >= WEIGHT_CAP).astype(int)
    out["weight"] = out["weight"].fillna(out["equipment"].map(weight_fill))
    return out


def add_date_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add date-level market aggregates.

    ``frame`` must hold every row whose features may be pooled (development rows
    and the rows to predict). Only *features* of rows sharing the same calendar
    date are pooled - no labels and no information from other dates - so these
    are available at prediction time and cannot leak the target.
    """
    out = frame.copy()
    by_date = out.groupby("date")
    out["mi_date"] = by_date["market_index"].transform("median")
    out["market_index_filled"] = out["market_index"].fillna(out["mi_date"])
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["qs_date"] = by_date["quote_signal"].transform("mean")
    out["qs_date_equip"] = out.groupby(["date", "equipment"])["quote_signal"].transform("mean")
    return out


def prepare(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Clean both frames consistently and add date context."""
    weight_fill = fit_weight_fill(train)
    train_c = clean_inputs(train, weight_fill)
    valid_c = clean_inputs(valid, weight_fill)
    pooled = add_date_context(pd.concat([train_c.assign(_src="train"), valid_c.assign(_src="valid")], ignore_index=True))
    train_c = pooled[pooled["_src"] == "train"].drop(columns="_src").reset_index(drop=True)
    valid_c = pooled[pooled["_src"] == "valid"].drop(columns="_src").reset_index(drop=True)
    return train_c, valid_c, weight_fill


def _design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log(frame["distance"]),
            frame["market_index_filled"],
            np.log(frame["market_index_filled"]),
            frame["weight"] / 1e4,
            (frame["equipment"] == "Reefer").astype(float),
            (frame["equipment"] == "Flatbed").astype(float),
            np.ones(len(frame)),
        ]
    )


def flag_label_outliers(train: pd.DataFrame, sigma: float = OUTLIER_SIGMA, n_iter: int = 3) -> pd.DataFrame:
    """Flag corrupted ``posted_rate`` labels with an iteratively re-fitted robust rule.

    A log-linear model on distance, equipment, weight and market index explains
    ~99.5% of the variance of clean rows. Rows whose residual exceeds ``sigma``
    robust standard deviations (MAD based) are flagged. The residual
    distribution has an empty gap (largest normal residual 0.18 log units,
    smallest flagged one 0.80), so the flag is insensitive to the exact
    threshold anywhere in roughly 5-15 sigma.

    Returns a frame with ``log_resid`` and ``label_outlier`` (bool) columns.
    """
    design = _design(train)
    y = np.log(train[TARGET].to_numpy())
    use = np.ones(len(train), dtype=bool)
    for _ in range(n_iter):
        beta, *_ = np.linalg.lstsq(design[use], y[use], rcond=None)
        resid = y - design @ beta
        mad_sigma = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use])))
        use = np.abs(resid) <= sigma * mad_sigma
    out = train.copy()
    out["log_resid"] = resid
    out["label_outlier"] = ~use
    return out
