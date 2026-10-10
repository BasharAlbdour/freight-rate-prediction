"""Train the final model on all labeled data and write the deliverable predictions.

Outputs
    validation_predictions.csv      12,000 rows: load_id,predicted_rate
    data/december_chart_inputs.csv  the provided file with predicted_rate filled in
    reports/final_model_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freight_rate import chart as C  # noqa: E402
from freight_rate import data as D  # noqa: E402
from freight_rate import features as F  # noqa: E402
from freight_rate import models as M  # noqa: E402

REPORTS = ROOT / "reports"
PARAMS_FILE = REPORTS / "best_boost_params.json"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    params = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(M.DEFAULT_BOOST_PARAMS)

    train_raw, valid_raw = D.load_raw()
    train, valid, _ = D.prepare(train_raw, valid_raw)
    train = D.flag_label_outliers(train)
    keep = ~train["label_outlier"].to_numpy()
    print(f"Training on {keep.sum():,} of {len(train):,} rows ({(~keep).sum():,} corrupted labels removed)")

    X_train, X_valid = F.build_features(train), F.build_features(valid)
    y = np.log(train[D.TARGET].to_numpy())
    model = M.build_final(params).fit(X_train[keep], y[keep])
    coef = dict(zip(F.LINEAR_FINAL, model.linear_.coef_))
    print("Linear part:", {k: round(float(v), 5) for k, v in coef.items()})
    print(f"Trend: {coef['t_days'] * 30 * 100:+.2f}% per 30 days | quarter-end ramp: {np.expm1(coef['qend_ramp']) * 100:+.2f}% at quarter end")

    # ---- 12,000 validation predictions, in template order --------------------------------
    template = pd.read_csv(ROOT / "data" / "validation_predictions_template.csv")
    pred = pd.Series(np.exp(model.predict(X_valid)), index=valid["load_id"]).round(2)
    out = pd.DataFrame({"load_id": template["load_id"], "predicted_rate": pred.reindex(template["load_id"]).to_numpy()})
    assert len(out) == 12_000 and out["load_id"].is_unique
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(out):,} rows)")

    # ---- December chart rows ---------------------------------------------------------------
    chart_path = ROOT / "data" / "december_chart_inputs.csv"
    chart = pd.read_csv(chart_path)
    chart_frame = C.build_chart_frame(chart, valid, C.city_coordinates(train_raw, valid_raw))
    chart["predicted_rate"] = np.exp(model.predict(F.build_features(chart_frame))).round(2)
    assert list(chart.columns) == C.CHART_COLUMNS and chart["predicted_rate"].notna().all()
    chart.to_csv(chart_path, index=False)
    print(f"Filled {chart_path.relative_to(ROOT)}")

    # ---- sanity checks used in the report --------------------------------------------------
    lane = train_raw[(train_raw.pickup == "Lexington") & (train_raw.delivery == "Fort Wayne") & (train_raw.equipment == "Dry Van")]
    lane = lane[~train.loc[lane.index, "label_outlier"]]
    summary = {
        "boost_params": params,
        "n_train_rows": int(len(train)),
        "n_train_rows_used": int(keep.sum()),
        "n_label_outliers_removed": int((~keep).sum()),
        "linear_coefficients": {k: float(v) for k, v in coef.items()},
        "linear_intercept": float(model.linear_.intercept_),
        "validation_pred_rate": out["predicted_rate"].describe().round(2).to_dict(),
        "train_posted_rate_clean": train.loc[keep, D.TARGET].describe().round(2).to_dict(),
        "december": {
            "min": float(chart.predicted_rate.min()),
            "max": float(chart.predicted_rate.max()),
            "mean": float(chart.predicted_rate.mean()),
            "per_mile_mean": float((chart.predicted_rate / chart.distance).mean()),
        },
        "lexington_fort_wayne_dry_van_history": {
            "n": int(len(lane)),
            "mean_rate": float(lane.posted_rate.mean()),
            "mean_rate_per_mile": float((lane.posted_rate / lane.distance).mean()),
            "last_month_in_train_mean": float(lane[lane.date >= "2025-10-01"].posted_rate.mean()) if (lane.date >= "2025-10-01").any() else None,
        },
    }
    (REPORTS / "final_model_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("december", "lexington_fort_wayne_dry_van_history", "validation_pred_rate")}, indent=2))


if __name__ == "__main__":
    main()
