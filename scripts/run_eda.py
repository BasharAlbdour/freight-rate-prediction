"""Exploratory analysis: data-quality counts (JSON) and the figures used in the report.

Writes reports/eda_summary.json and reports/fig_*.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freight_rate import data as D  # noqa: E402
from freight_rate import features as F  # noqa: E402
from freight_rate import validation as V  # noqa: E402

REPORTS = ROOT / "reports"

# Palette: first three slots of the reference categorical palette, neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.edgecolor": "#b9b8b1",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "legend.frameon": False,
})


def summarise_quality(train_raw: pd.DataFrame, valid_raw: pd.DataFrame, flagged: pd.DataFrame) -> dict:
    new_cities = sorted((set(valid_raw.pickup) | set(valid_raw.delivery)) - (set(train_raw.pickup) | set(train_raw.delivery)))
    touches = valid_raw.pickup.isin(new_cities) | valid_raw.delivery.isin(new_cities)
    out = flagged[flagged.label_outlier]
    ratio = np.exp(out.log_resid)
    daily = flagged[~flagged.label_outlier].groupby("date").log_resid.mean()
    return {
        "rows": {"train_test": len(train_raw), "validation": len(valid_raw)},
        "date_range": {
            "train_test": [str(train_raw.date.min().date()), str(train_raw.date.max().date())],
            "validation": [str(valid_raw.date.min().date()), str(valid_raw.date.max().date())],
        },
        "duplicates": {"load_id": int(train_raw.load_id.duplicated().sum()), "full_rows": int(train_raw.drop(columns="load_id").duplicated().sum())},
        "weight": {
            "negative_train": int((train_raw.weight < 0).sum()), "negative_validation": int((valid_raw.weight < 0).sum()),
            "missing_train": int(train_raw.weight.isna().sum()), "missing_validation": int(valid_raw.weight.isna().sum()),
            "at_cap_47500_train": int((train_raw.weight.abs() >= D.WEIGHT_CAP).sum()),
        },
        "market_index_missing": {"train": int(train_raw.market_index.isna().sum()), "validation": int(valid_raw.market_index.isna().sum())},
        "label_outliers": {
            "flagged": int(len(out)), "share_%": round(100 * len(out) / len(flagged), 2),
            "too_low_(<1x_expected)": int((ratio < 1).sum()), "too_high_(>1x_expected)": int((ratio > 1).sum()),
            "ratio_to_expected_low_range": [round(float(ratio[ratio < 1].min()), 2), round(float(ratio[ratio < 1].max()), 2)],
            "ratio_to_expected_high_range": [round(float(ratio[ratio > 1].min()), 2), round(float(ratio[ratio > 1].max()), 2)],
            "largest_clean_abs_log_resid": round(float(flagged[~flagged.label_outlier].log_resid.abs().max()), 3),
            "smallest_flagged_abs_log_resid": round(float(out.log_resid.abs().min()), 3),
        },
        "cities_only_in_validation": new_cities,
        "validation_rows_touching_new_cities": int(touches.sum()),
        "lanes": {"train": int(train_raw.groupby(["pickup", "delivery"]).ngroups), "validation_lanes_unseen_in_train": int(
            (~valid_raw.set_index(["pickup", "delivery"]).index.isin(train_raw.set_index(["pickup", "delivery"]).index)).sum())},
        "daily_latent_state": {
            "sd_of_daily_mean_log_residual": round(float(daily.std()), 4),
            "lag1_autocorr": round(float(daily.autocorr(1)), 3),
            "lag7_autocorr": round(float(daily.autocorr(7)), 3),
            "lag30_autocorr": round(float(daily.autocorr(30)), 3),
        },
    }


def fig_split(train_raw: pd.DataFrame, valid_raw: pd.DataFrame) -> None:
    both = pd.concat([train_raw, valid_raw])
    daily_mi = both.groupby("date").market_index.median()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.8), gridspec_kw={"height_ratios": [2.2, 1.5], "hspace": 0.55})
    ax.plot(daily_mi.index, daily_mi.values, color=BLUE, lw=1.2)
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=ORANGE, alpha=0.12, lw=0)
    ax.text(pd.Timestamp("2025-01-08"), 1.5, "Labeled development data (Jan-Oct)", color=BLUE, fontsize=9, ha="left")
    ax.text(pd.Timestamp("2025-12-01"), 1.28, "Validation:\nloads to\npredict\n(Nov-Dec)", color=ORANGE, fontsize=9, ha="center")
    ax.set_ylabel("Daily median market_index")
    ax.set_ylim(0.65, 1.58)
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("The validation set is the future, so the split has to be chronological")

    folds = [("Fold 1", "2025-01-01", "2025-05-01", "2025-07-01"), ("Fold 2", "2025-01-01", "2025-07-01", "2025-09-01"),
             ("Fold 3", "2025-01-01", "2025-09-01", "2025-11-01"), ("Final", "2025-01-01", "2025-11-01", "2026-01-01")]
    for i, (name, start, mid, end) in enumerate(folds[::-1]):
        start, mid, end = map(pd.Timestamp, (start, mid, end))
        final = name == "Final"
        bx.barh(i, (mid - start).days, left=start, height=0.58, color=BLUE, lw=0)
        bx.barh(i, (end - mid).days, left=mid, height=0.58, color=ORANGE if final else AQUA, lw=0)
        bx.text(start + (mid - start) / 2, i, "train" if not final else "train on all labeled data", color="white", ha="center", va="center", fontsize=8.5)
        bx.text(mid + (end - mid) / 2, i, "predict" if final else "test", color="white", ha="center", va="center", fontsize=8.5)
    bx.set_yticks(range(4))
    bx.set_yticklabels([f[0] for f in folds[::-1]])
    bx.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))
    bx.xaxis.set_major_locator(mdates.MonthLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    bx.grid(axis="y", visible=False)
    bx.set_title("Forward-chaining validation: train on the past, test on the next two months")
    fig.savefig(REPORTS / "fig_split.png", bbox_inches="tight")
    plt.close(fig)


def fig_outliers(flagged: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    clipped = flagged.log_resid.clip(-2, 2)
    bins = np.linspace(-2, 2, 81)
    ax.hist(clipped[~flagged.label_outlier], bins=bins, color=BLUE, lw=0)
    ax.hist(clipped[flagged.label_outlier], bins=bins, color=ORANGE, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e4)
    ax.set_xlabel("log(posted_rate) minus the rate implied by the load's features  (0 = as expected)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("677 labels (1.4%) are corrupted: a clean gap separates them from normal noise")
    ax.text(0.28, 1.2e4, "normal rows (residual sd ~0.04)", color=BLUE, ha="left", fontsize=9)
    ax.text(-1.25, 900, "0.17-0.45x too low", color=ORANGE, ha="center", fontsize=9)
    ax.text(1.15, 900, "2.2-5.4x too high", color=ORANGE, ha="center", fontsize=9)
    fig.savefig(REPORTS / "fig_label_outliers.png", bbox_inches="tight")
    plt.close(fig)


def fig_latent_state(train_prepared: pd.DataFrame, flagged: pd.DataFrame, valid_raw: pd.DataFrame, train_raw: pd.DataFrame) -> None:
    clean = flagged[~flagged.label_outlier]
    feats = F.build_features(clean)
    daily = (
        clean.assign(t=feats["t_days"], ramp=feats["qend_ramp"])
        .groupby("date").agg(res=("log_resid", "mean"), t=("t", "first"), ramp=("ramp", "first")).reset_index()
    )
    design = np.column_stack([np.ones(len(daily)), daily.t, daily.ramp])
    beta, *_ = np.linalg.lstsq(design, daily.res.to_numpy(), rcond=None)

    future = pd.DataFrame({"date": pd.date_range("2025-11-01", "2025-12-31")})
    future_feats = F.build_features(future.assign(**{c: 0.0 for c in ()}).assign(
        distance=1.0, weight=1.0, equipment="Dry Van", market_index_filled=1.0, mi_date=1.0, quote_signal=1.0,
        qs_date=1.0, qs_date_equip=1.0, pickup_lat=0.0, pickup_lon=0.0, delivery_lat=0.0, delivery_lon=0.0))
    future_fit = np.column_stack([np.ones(len(future)), future_feats["t_days"], future_feats["qend_ramp"]]) @ beta

    both = pd.concat([train_raw, valid_raw])
    dry = both[both.equipment == "Dry Van"].groupby("date").quote_signal.mean()

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"hspace": 0.32})
    ax.plot(daily.date, daily.res * 100, color=BLUE, lw=1.2, label="Observed daily mean residual")
    ax.plot(daily.date, (design @ beta) * 100, color=ORANGE, lw=1.4, label="Trend + quarter-end ramp (fitted)")
    ax.plot(future.date, future_fit * 100, color=ORANGE, lw=1.4, ls=(0, (3, 2)), label="Same pattern carried into Nov-Dec")
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=ORANGE, alpha=0.10, lw=0)
    ax.axhline(0, color="#b9b8b1", lw=0.8)
    ax.set_ylabel("Rate vs. feature-implied (%)")
    ax.set_title("Beyond the load's own features, rates follow a calendar pattern")
    ax.legend(loc="upper left", fontsize=8.5)
    bx.plot(dry.index, dry.values, color=AQUA, lw=1.2)
    bx.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=ORANGE, alpha=0.10, lw=0)
    bx.set_ylabel("Mean quote_signal\n(Dry Van, daily)")
    bx.set_title("Date-level quote_signal averages out row noise and partly tracks the market state")
    bx.xaxis.set_major_locator(mdates.MonthLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.savefig(REPORTS / "fig_latent_state.png", bbox_inches="tight")
    plt.close(fig)


def fig_models() -> None:
    summary = pd.read_csv(REPORTS / "cv_summary.csv")
    clean = summary[summary.subset == "clean"]
    fwd_label = "forward-chaining"
    fwd = clean[clean.scheme == fwd_label].set_index("model")["MAE"]
    rnd = clean[clean.scheme.str.startswith("random")].set_index("model")["MAE"]
    names = [m for m in fwd.index]
    short = [n.split(". ", 1)[1].replace("Final: ", "FINAL: ").replace(" (", "\n(", 1) if len(n) > 46 else n.split(". ", 1)[1] for n in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.barh(y + 0.19, fwd.values, height=0.34, color=BLUE, lw=0, label="Forward-chaining (matches the real task)")
    rnd_vals = [rnd.get(n, np.nan) for n in names]
    ax.barh(y - 0.19, rnd_vals, height=0.34, color=ORANGE, lw=0, label="Random 5-fold (leaks neighbouring days: too optimistic)")
    for yy, v in zip(y + 0.19, fwd.values):
        ax.text(v + 1.5, yy, f"${v:.0f}", va="center", fontsize=8.5, color=INK)
    for yy, v in zip(y - 0.19, rnd_vals):
        if not np.isnan(v):
            ax.text(v + 1.5, yy, f"${v:.0f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Mean absolute error ($ per load, clean labels, lower is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(0, 122)
    fig.savefig(REPORTS / "fig_models.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    train_raw, valid_raw = D.load_raw()
    train, _, _ = D.prepare(train_raw, valid_raw)
    flagged = D.flag_label_outliers(train)
    (REPORTS / "eda_summary.json").write_text(json.dumps(summarise_quality(train_raw, valid_raw, flagged), indent=2))
    fig_split(train_raw, valid_raw)
    fig_outliers(flagged)
    fig_latent_state(train, flagged, valid_raw, train_raw)
    if (REPORTS / "cv_summary.csv").exists():
        fig_models()
    print(json.dumps(json.loads((REPORTS / "eda_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
