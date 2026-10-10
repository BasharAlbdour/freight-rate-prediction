"""Feature construction and feature groups used by the models.

Design constraints
* Eight cities occur only in the validation file, so no feature may identify a
  city. Geography enters through coordinates, which are consistent per city and
  let the model generalise to unseen origins/destinations.
* The validation period (Nov-Dec) lies *after* all labeled data, so the part of
  the model that carries calendar structure must extrapolate. Trees cannot (they
  hold the last value), hence trend and quarter-end effects live in the linear part.

Calendar structure found in the labels (see reports/fig_latent_state.png): after
removing the effect of the load's own features, the daily mean log-rate follows a
linear upward trend (~+0.6% per 30 days) plus a ramp over the last ~4 weeks of
every quarter (~+4% on the final day) that resets on the first day of the next
quarter. Together they explain the daily residual down to ~0.57% sd (pure noise
is ~0.31%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_ORIGIN = pd.Timestamp("2025-01-01")
QUARTER_END_RAMP_DAYS = 28

# Linear (structural) part: rate scales with distance, equipment premium, market level, calendar.
LINEAR_BASE = ["log_distance", "is_reefer", "is_flatbed"]
LINEAR_CALENDAR = ["t_days", "qend_ramp"]
LINEAR_STRUCTURE = LINEAR_BASE + ["mi"]  # without calendar terms
LINEAR_FINAL = LINEAR_STRUCTURE + LINEAR_CALENDAR
LINEAR_ALL = LINEAR_FINAL + ["weight_10k", "quote_signal", "qs_date_equip"]

# Boosted part. ``mi_date`` / ``qs_date*`` are same-day aggregates of other loads'
# features: a low-noise read-out of the smooth latent market state.
GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
BOOST_CORE = ["log_distance", "weight", "mi", "quote_signal", "is_reefer", "is_flatbed"]
DATE_CONTEXT = ["mi_date", "qs_date", "qs_date_equip"]
BOOST_ALL = BOOST_CORE + GEO + DATE_CONTEXT


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a prepared frame (see ``data.prepare``) into the model feature matrix."""
    feats = pd.DataFrame(index=frame.index)
    feats["log_distance"] = np.log(frame["distance"])
    feats["weight"] = frame["weight"]
    feats["weight_10k"] = frame["weight"] / 1e4
    feats["is_reefer"] = (frame["equipment"] == "Reefer").astype(float)
    feats["is_flatbed"] = (frame["equipment"] == "Flatbed").astype(float)
    feats["mi"] = frame["market_index_filled"]
    feats["mi_date"] = frame["mi_date"]
    feats["quote_signal"] = frame["quote_signal"]
    feats["qs_date"] = frame["qs_date"]
    feats["qs_date_equip"] = frame["qs_date_equip"]
    for col in GEO:
        feats[col] = frame[col]

    # Calendar terms.
    quarter = frame["date"].dt.to_period("Q")
    days_to_quarter_end = (quarter.dt.end_time.dt.normalize() - frame["date"]).dt.days.astype(float)
    feats["t_days"] = (frame["date"] - TIME_ORIGIN).dt.days.astype(float)
    feats["days_to_quarter_end"] = days_to_quarter_end
    feats["qend_ramp"] = np.clip(QUARTER_END_RAMP_DAYS - days_to_quarter_end, 0, QUARTER_END_RAMP_DAYS) / QUARTER_END_RAMP_DAYS
    return feats
