"""Compare candidate models under forward-chaining, unseen-city and random CV.

Writes CSV tables to reports/ and prints the numbers quoted in the report.
Run time: roughly ten minutes on a laptop.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freight_rate import data as D  # noqa: E402
from freight_rate import features as F  # noqa: E402
from freight_rate import models as M  # noqa: E402
from freight_rate import validation as V  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)

    train_raw, valid_raw = D.load_raw()
    train, _, _ = D.prepare(train_raw, valid_raw)
    train = D.flag_label_outliers(train)
    X = F.build_features(train)
    print(f"Development rows: {len(train):,} | flagged label outliers: {int(train.label_outlier.sum()):,}")

    # --- 1. small hyper-parameter search, scored by forward-chaining on clean rows ---------
    rows = []
    for leaves, iters in product([15, 31, 63], [300, 600]):
        params = dict(M.DEFAULT_BOOST_PARAMS, max_leaf_nodes=leaves, max_iter=iters)
        res = V.forward_chaining({"final": lambda p=params: M.build_final(p)}, train, X)
        clean = res[res.subset == "clean"]
        rows.append({"max_leaf_nodes": leaves, "max_iter": iters, "MAE_clean": clean.MAE.mean(), "MAPE_clean_%": clean["MAPE_%"].mean()})
    tuning = pd.DataFrame(rows).round(3).sort_values("MAE_clean").reset_index(drop=True)
    print("\nHyper-parameter search (forward-chaining, mean over 3 folds, clean labels):")
    print(tuning.to_string(index=False))
    tuning.to_csv(REPORTS / "hyperparameter_search.csv", index=False)
    best = tuning.iloc[0]
    best_params = dict(M.DEFAULT_BOOST_PARAMS, max_leaf_nodes=int(best.max_leaf_nodes), max_iter=int(best.max_iter))
    (REPORTS / "best_boost_params.json").write_text(json.dumps(best_params, indent=2))
    print("Chosen boosting parameters:", best_params)

    # --- 2. model comparison under the three schemes ---------------------------------------
    models = M.candidate_models(best_params)
    results = pd.concat(
        [V.forward_chaining(models, train, X), V.unseen_cities(models, train, X), V.random_kfold(models, train, X)],
        ignore_index=True,
    )
    results.to_csv(REPORTS / "cv_results.csv", index=False)
    summary = pd.concat([V.summarise(results, s).assign(subset=s) for s in ("clean", "raw")])
    summary.to_csv(REPORTS / "cv_summary.csv", index=False)
    for subset in ("clean", "raw"):
        print(f"\n=== Mean metrics, {subset} labels ===")
        print(summary[summary.subset == subset].drop(columns="subset").to_string(index=False))

    print("\n=== Forward-chaining per fold, final model ===")
    for subset in ("clean", "raw"):
        per_fold = results[(results.model == M.FINAL_MODEL_NAME) & (results.scheme == "forward-chaining") & (results.subset == subset)]
        print(f"[{subset} labels]")
        print(per_fold[["fold", "n_test", "MAE", "RMSE", "MAPE_%", "R2"]].round(3).to_string(index=False))
        per_fold.to_csv(REPORTS / f"final_per_fold_{subset}.csv", index=False)

    # --- 3. ramp-length sensitivity (the one hand-set calendar constant) --------------------
    print("\n=== Quarter-end ramp length (forward-chaining, clean) ===")
    rows = []
    for k in (14, 21, 28, 35, 42):
        Xk = X.copy()
        Xk["qend_ramp"] = np.clip(k - Xk["days_to_quarter_end"], 0, k) / k
        res = V.forward_chaining({"final": models[M.FINAL_MODEL_NAME]}, train, Xk)
        clean = res[res.subset == "clean"]
        rows.append({"ramp_days": k, "MAE": clean.MAE.mean(), "MAPE_%": clean["MAPE_%"].mean(), **{f"MAE {f}": v for f, v in zip(clean.fold, clean.MAE)}})
    ramp = pd.DataFrame(rows).round(2)
    print(ramp.to_string(index=False))
    ramp.to_csv(REPORTS / "ramp_sensitivity.csv", index=False)

    # --- 4. does carrying the recent market level forward still help? ----------------------
    print("\n=== Recent-level carry-over, final model (forward-chaining, clean) ===")
    rows = []
    for window in (None, 7, 14, 28):
        res = V.forward_chaining({"final": models[M.FINAL_MODEL_NAME]}, train, X, level_window_days=window)
        clean = res[res.subset == "clean"]
        rows.append({"window_days": "none" if window is None else window, "MAE": clean.MAE.mean(), "MAPE_%": clean["MAPE_%"].mean()})
    carry = pd.DataFrame(rows).round(3)
    print(carry.to_string(index=False))
    carry.to_csv(REPORTS / "level_carryover.csv", index=False)

    # --- 5. feature-group ablation of the final model --------------------------------------
    print("\n=== Ablation of the final model (forward-chaining, clean labels) ===")

    def without(*drop: str, linear: list[str] | None = None):
        cols = [c for c in F.BOOST_ALL if c not in drop]
        return lambda: M.build_final(best_params, boost_cols=cols, linear_cols=linear)

    ablations = {
        "final (all features)": lambda: M.build_final(best_params),
        "- quarter-end ramp (keep trend)": without(linear=F.LINEAR_STRUCTURE + ["t_days"]),
        "- trend (keep quarter-end ramp)": without(linear=F.LINEAR_STRUCTURE + ["qend_ramp"]),
        "- coordinates": without(*F.GEO),
        "- date-level quote_signal": without("qs_date", "qs_date_equip"),
        "- date-level market_index": without("mi_date"),
        "- row-level quote_signal": without("quote_signal"),
        "- weight": without("weight"),
    }
    ab = V.summarise(V.forward_chaining(ablations, train, X), "clean").drop(columns="scheme")
    print(ab.to_string(index=False))
    ab.to_csv(REPORTS / "ablation.csv", index=False)


if __name__ == "__main__":
    main()
