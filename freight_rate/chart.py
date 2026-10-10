"""Features for the fixed December 2025 chart rows.

``december_chart_inputs.csv`` only carries the seven lane/equipment/date columns,
but the model also uses market signals. For each December date they are taken
from the *other* December loads in ``validation.csv`` (features only, never
labels): the same pooling used for every other row, applied at the calendar-day
and equipment level.

* ``market_index``  -> that day's median market_index (the row-level value is the
  day level plus small noise, so the day level is its expectation)
* ``quote_signal``  -> that day's mean quote_signal for the equipment type
"""

from __future__ import annotations

import pandas as pd

CHART_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city; asserts the coordinates are consistent across files."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(parts).drop_duplicates()
    if coords["city"].duplicated().any():
        raise ValueError("a city has more than one coordinate pair")
    return coords.set_index("city")


def build_chart_frame(chart: pd.DataFrame, valid_prepared: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the columns ``features.build_features`` expects."""
    frame = chart.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["distance"] = frame["distance"].astype(float)
    frame["weight"] = frame["weight"].astype(float)

    for role in ("pickup", "delivery"):
        looked_up = coords.reindex(frame[role])
        if looked_up.isna().any().any():
            raise ValueError(f"unknown {role} city in chart inputs")
        frame[f"{role}_lat"] = looked_up["lat"].to_numpy()
        frame[f"{role}_lon"] = looked_up["lon"].to_numpy()

    day = valid_prepared.groupby("date").agg(mi_date=("mi_date", "first"), qs_date=("qs_date", "first"))
    day_equip = valid_prepared.groupby(["date", "equipment"])["qs_date_equip"].first().rename("qs_date_equip").reset_index()
    frame = frame.merge(day, left_on="date", right_index=True, how="left")
    frame = frame.merge(day_equip, on=["date", "equipment"], how="left")
    if frame[["mi_date", "qs_date", "qs_date_equip"]].isna().any().any():
        raise ValueError("a chart date/equipment has no validation loads to pool market signals from")

    frame["market_index_filled"] = frame["mi_date"]
    frame["quote_signal"] = frame["qs_date_equip"]
    return frame
