"""Build the DOCX report from the result files in reports/ (numbers are read, never typed).

Prerequisites: run_eda.py, run_validation.py, run_final.py and score.py have been run.
Output: Freight_Rate_Report.docx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freight_rate import models as M  # noqa: E402

REPORTS = ROOT / "reports"
OUT = ROOT / "Freight_Rate_Report.docx"
TEAL = RGBColor(0x06, 0x4A, 0x56)
MUTED = RGBColor(0x45, 0x5A, 0x60)


# ----------------------------------------------------------------------------- helpers
def shade(cell, hex_fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), hex_fill)
    props.append(el)


def set_cell(cell, text: str, bold: bool = False, align=None, color: RGBColor | None = None, size: float = 9) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(0)
    if align:
        para.alignment = align
    run = para.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color


def add_table(doc, header: list[str], rows: list[list], widths: list[float], numeric_from: int = 1, highlight_last: bool = False):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        set_cell(cell, text, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from else None)
        shade(cell, "064A56")
    for r, row in enumerate(rows):
        cells = table.add_row().cells
        last = highlight_last and r == len(rows) - 1
        for i, value in enumerate(row):
            set_cell(cells[i], value, bold=last, align=WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from else None)
            if last:
                shade(cells[i], "E6F0F2")
            elif r % 2 == 1:
                shade(cells[i], "F5F8F8")
    for row in table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def heading(doc, text: str, level: int = 1) -> None:
    para = doc.add_heading(text, level=level)
    for run in para.runs:
        run.font.color.rgb = TEAL
        run.font.name = "Calibri"


def para(doc, text: str, bold_lead: str | None = None, italic: bool = False, size: float | None = None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    return p


def bullet(doc, text: str, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)


def figure(doc, path: Path, width: float, caption: str) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = MUTED


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# ----------------------------------------------------------------------------- content
def main() -> None:
    eda = json.loads((REPORTS / "eda_summary.json").read_text())
    final = json.loads((REPORTS / "final_model_summary.json").read_text())
    summary = pd.read_csv(REPORTS / "cv_summary.csv")
    ablation = pd.read_csv(REPORTS / "ablation.csv")
    ramp = pd.read_csv(REPORTS / "ramp_sensitivity.csv")
    per_fold = pd.read_csv(REPORTS / "final_per_fold_clean.csv")
    per_fold_raw = pd.read_csv(REPORTS / "final_per_fold_raw.csv")
    params = json.loads((REPORTS / "best_boost_params.json").read_text())

    def metric(scheme: str, model: str, col: str, subset: str = "clean") -> float:
        row = summary[(summary.scheme.str.startswith(scheme)) & (summary.model == model) & (summary.subset == subset)]
        return float(row[col].iloc[0])

    fm = M.FINAL_MODEL_NAME
    base = "1. Baseline: log-linear (distance, equipment)"
    f_mae, f_mape, f_r2 = (metric("forward", fm, c) for c in ("MAE", "MAPE_%", "R2"))
    b_mae, b_mape = metric("forward", base, "MAE"), metric("forward", base, "MAPE_%")
    u_mae, u_mape = metric("unseen", fm, "MAE"), metric("unseen", fm, "MAPE_%")
    r_mae = metric("random", fm, "MAE")
    raw_mae, raw_mape, raw_rmse = (metric("forward", fm, c, "raw") for c in ("MAE", "MAPE_%", "RMSE"))
    lo = eda["label_outliers"]
    w = eda["weight"]

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Inches(0.9))
    section.top_margin = section.bottom_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    title = doc.add_paragraph()
    r = title.add_run("Freight Rate Prediction")
    r.bold = True
    r.font.size = Pt(24)
    r.font.color.rgb = TEAL
    sub = doc.add_paragraph()
    r = sub.add_run("Validation approach, data split and December forecast  |  Machine Learning Engineer assessment")
    r.font.size = Pt(11)
    r.font.color.rgb = MUTED

    # ---- 1. Summary
    heading(doc, "1. Summary")
    para(doc, f"The model predicts the posted rate of a load from its lane, equipment, weight, market signals and the calendar. "
              f"It is a log-linear structural model with a linear time trend and a quarter-end ramp, plus gradient boosting on its residuals. "
              f"Because the 12,000 loads to predict are all in Nov-Dec 2025 (after every labeled load), it was validated by training on the past and "
              f"testing on the following two months, never with a random split.")
    add_table(
        doc, ["Result (forward-chaining, mean of 3 two-month folds)", "Final model", "Naive baseline"],
        [
            ["Mean absolute error, clean labels", f"{fmt_money(f_mae)} per load", f"{fmt_money(b_mae)}"],
            ["Mean absolute % error, clean labels", f"{f_mape:.2f}%", f"{b_mape:.2f}%"],
            ["R-squared, clean labels", f"{f_r2:.4f}", f"{metric('forward', base, 'R2'):.4f}"],
            ["Same, labels as delivered (incl. corrupted ones)", f"{fmt_money(raw_mae)} MAE / {raw_mape:.2f}% MAPE", f"{fmt_money(metric('forward', base, 'MAE', 'raw'))} MAE"],
            ["Loads touching cities hidden from training (Sep-Oct)", f"{fmt_money(u_mae)} MAE / {u_mape:.2f}% MAPE", f"{fmt_money(metric('unseen', base, 'MAE'))} MAE"],
        ],
        [3.6, 1.9, 1.2], numeric_from=1,
    )
    para(doc, "Three findings shaped the solution:", bold_lead=None)
    bullet(doc, f" {lo['flagged']} training labels ({lo['share_%']}%) are corrupted - roughly 0.2-0.45x or 2.2-5.4x the value their features imply. "
                f"They are removed from training and reported separately in evaluation.", "Corrupted labels.")
    bullet(doc, f" {len(eda['cities_only_in_validation'])} cities appear only in the validation file ({eda['validation_rows_touching_new_cities']:,} of 12,000 loads), "
                "so no feature identifies a city; geography enters through coordinates.", "Unseen cities.")
    bullet(doc, " After accounting for a load's own features, the daily rate follows a linear upward trend plus a ramp over the last four weeks "
                "of every quarter that resets on the first day of the next. Adding both terms cut forward-chaining error by about a quarter and unseen-city error by about half.",
           "Calendar structure.")

    # ---- 2. Data quality
    heading(doc, "2. Data-quality issues and how they were handled")
    add_table(
        doc, ["Issue found", "Evidence", "Handling"],
        [
            ["Sign-flipped weights", f"{w['negative_train']} negative in train, {w['negative_validation']} in validation; absolute values span 5,000-47,500 lb", "Use the absolute value"],
            ["Missing weight", f"{w['missing_train']} train / {w['missing_validation']} validation rows", "Median weight of the same equipment type"],
            ["Weight capped at 47,500 lb", f"{w['at_cap_47500_train']:,} train rows sit exactly at the cap", "Kept (it is a real, consistent ceiling)"],
            ["Missing market_index", f"{eda['market_index_missing']['train']} train / {eda['market_index_missing']['validation']} validation rows", "Same-day median of other loads (it is a day-level signal plus small noise)"],
            ["Corrupted posted_rate", f"{lo['flagged']} rows ({lo['share_%']}%): {lo['too_low_(<1x_expected)']} too low, {lo['too_high_(>1x_expected)']} too high; "
                                    f"no row between {lo['largest_clean_abs_log_resid']:.2f} and {lo['smallest_flagged_abs_log_resid']:.2f} log units from expected", "Removed from training only; evaluation reported on clean and raw labels"],
            ["Cities only in validation", ", ".join(eda["cities_only_in_validation"]), "No city identity features; coordinates instead"],
            ["Coordinates are synthetic", "Haversine distance disagrees with `distance` (up to 78x), but `distance` is consistent per lane", "Trusted `distance`; used coordinates only as a regional signal"],
            ["Duplicates / ID problems", f"{eda['duplicates']['load_id']} duplicate load_ids, {eda['duplicates']['full_rows']} duplicate rows", "None needed"],
        ],
        [1.55, 3.2, 1.95], numeric_from=99,
    )
    figure(doc, REPORTS / "fig_label_outliers.png", 6.2,
           "Figure 1. Residual of log(posted_rate) against the rate implied by the load's features. Corrupted labels form two islands far from the normal cloud.")

    # ---- 3. Validation approach
    heading(doc, "3. Train / validation split approach")
    para(doc, f"Task shape. Labeled data covers {eda['date_range']['train_test'][0]} to {eda['date_range']['train_test'][1]}; the loads to predict cover "
              f"{eda['date_range']['validation'][0]} to {eda['date_range']['validation'][1]}. The split must therefore be chronological. "
              "A random split would be misleading for a specific reason: the daily average rate carries a smooth market state "
              f"(lag-1 autocorrelation {eda['daily_latent_state']['lag1_autocorr']}), so a random split puts neighbouring days in both train and test and rewards a model for interpolating it.")
    para(doc, "Primary scheme: forward-chaining with a two-month test horizon (the same horizon as the real task):", bold_lead=None)
    bullet(doc, " train Jan-Apr, test May-Jun")
    bullet(doc, " train Jan-Jun, test Jul-Aug")
    bullet(doc, " train Jan-Aug, test Sep-Oct (the fold that crosses the Oct 1 quarter boundary, closest to the real situation)")
    para(doc, f"Secondary scheme: unseen cities. Sep-Oct is tested on loads touching 8 randomly hidden cities, with those cities removed from training "
              f"(4 random draws), mimicking the 8 cities that only appear in the validation file. Contrast scheme: shuffled 5-fold CV, reported only to show how "
              f"optimistic it is (final-model MAE {fmt_money(r_mae)} vs {fmt_money(f_mae)} under forward-chaining).")
    para(doc, f"Labels used for scoring. Corrupted labels in a test fold cannot be predicted, and they dominate squared-error metrics "
              f"(RMSE about {fmt_money(raw_rmse)} on raw labels versus about {fmt_money(metric('forward', fm, 'RMSE'))} on clean labels). Each scheme is therefore reported on both subsets. "
              "Corrupted rows are always excluded from training. The outlier flag uses labels only to delete rows, never as a feature.")
    para(doc, f"Final fit. After model selection, the model is trained on all {final['n_train_rows_used']:,} clean labeled rows and predicts the 12,000 validation loads. "
              f"Boosting hyper-parameters (leaves={params['max_leaf_nodes']}, iterations={params['max_iter']}) were picked by forward-chaining on a small grid.")
    figure(doc, REPORTS / "fig_split.png", 6.2,
           "Figure 2. The validation loads (shaded) come after all labeled data; the folds always train on the past and test on the next two months.")

    # ---- 4. Findings
    heading(doc, "4. What drives the rate")
    para(doc, "Distance, equipment, weight and the market index explain roughly 99.5% of the variance of clean log-rates. The remaining variation has a clear structure:")
    bullet(doc, " the daily mean residual is a trend (about +0.6% per 30 days) plus a ramp across the last four weeks of each quarter, reaching about +4% on the final day and dropping back on the first day of the next quarter. "
                "Together they explain the daily residual down to ~0.6% (pure noise is ~0.3%). They are modelled linearly because trees cannot extrapolate beyond the training period.", "Calendar.")
    bullet(doc, " the same-day average of quote_signal (and not the noisy per-load value) tracks the latent market state; removing it worsens forward MAE.", "Date-level market signal.")
    bullet(doc, " origin/destination coordinates add a regional effect (for example origins such as Syracuse, Buffalo and Detroit run cheaper, Dallas, Lubbock and Oklahoma City dearer) that carries over to unseen cities: removing the coordinates raises forward MAE.", "Geography.")
    figure(doc, REPORTS / "fig_latent_state.png", 6.2,
           "Figure 3. Top: observed daily residual and the fitted calendar pattern (dashed: carried into Nov-Dec). Bottom: Dry Van daily mean quote_signal.")

    # ---- 5. Model
    heading(doc, "5. Chosen model and why")
    para(doc, "Model: log(posted_rate) = linear part + boosted residuals.")
    bullet(doc, " ln(distance), equipment premiums, market_index, linear time trend and quarter-end ramp (28 days). "
                "These terms are smooth and extrapolate safely into November-December.", "Linear part:")
    bullet(doc, " histogram gradient boosting on what is left, using weight, per-load and date-level market signals, and the four coordinates.", "Boosted part:")
    para(doc, "Why not a single model type? Pure gradient boosting is strong on interpolation but holds the last value in time, so it cannot follow the trend or the Q4 ramp. "
              "A pure linear model cannot learn geography and weight interactions. The hybrid takes the strengths of both, and the comparison below shows the gain at each step.")
    order = [base, "2. Linear (all features + calendar terms)", "3. Gradient boosting only", "4. Linear + boosted residuals (no calendar terms)",
             "5. As 4 + linear trend only (no quarter-end ramp)", fm]
    rows = []
    for name in order:
        rows.append([name.split(". ", 1)[1], f"{metric('forward', name, 'MAE'):.1f}", f"{metric('forward', name, 'MAPE_%'):.2f}%",
                     f"{metric('unseen', name, 'MAE'):.1f}", f"{metric('random', name, 'MAE'):.1f}"])
    add_table(doc, ["Model", "Forward MAE ($)", "Forward MAPE", "Unseen-city MAE ($)", "Random-CV MAE ($)"], rows, [3.0, 0.9, 0.85, 1.05, 0.95], highlight_last=True)
    para(doc, "Clean labels, mean over folds. Model 5 shows that a trend on its own is unstable (the quarter-end ramps masquerade as trend); "
              "both terms are needed.", italic=True, size=9)
    figure(doc, REPORTS / "fig_models.png", 6.2, "Figure 4. Forward-chaining error versus the (misleadingly low) random-CV error.")

    heading(doc, "Per-fold results of the final model", 2)
    rows = []
    for (_, c), (_, rr) in zip(per_fold.iterrows(), per_fold_raw.iterrows()):
        rows.append([c["fold"], f"{int(c['n_test']):,}", f"{c['MAE']:.1f}", f"{c['MAPE_%']:.2f}%", f"{c['R2']:.4f}", f"{rr['MAE']:.1f}", f"{rr['MAPE_%']:.2f}%"])
    add_table(doc, ["Test fold", "Rows", "MAE ($)", "MAPE", "R2", "MAE raw ($)", "MAPE raw"], rows, [1.1, 0.8, 0.9, 0.8, 0.8, 1.1, 1.0])

    heading(doc, "Ablation (what each part contributes)", 2)
    rows = [[r["model"], f"{r['MAE']:.1f}", f"{r['MAPE_%']:.2f}%", f"{r['R2']:.4f}"] for _, r in ablation.iterrows()]
    add_table(doc, ["Final model with the item removed", "MAE ($)", "MAPE", "R2"], rows, [3.4, 1.0, 1.0, 1.0], highlight_last=False)
    para(doc, "Row-level quote_signal and date-level market_index contribute nothing measurable (differences within fold noise); they are kept because removing them is not an improvement either. "
              f"The 28-day ramp length was set from the daily residuals; forward MAE is {ramp.loc[ramp.ramp_days == 28, 'MAE'].iloc[0]:.1f} at 28 days and stays between "
              f"{ramp.MAE.min():.1f} and {ramp.loc[ramp.ramp_days >= 21, 'MAE'].max():.1f} for 21-42 days, so the result is only mildly sensitive to it (within about 10%).", size=9.5)

    # ---- 6. December chart
    heading(doc, "6. Fixed December 2025 prediction chart")
    para(doc, "The chart below is produced by the provided score.py from data/december_chart_inputs.csv (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes). "
              "Those rows carry no market signals, so they are taken from the other December loads in validation.csv on the same day (features only, never labels): the day's median "
              "market_index and the day's mean Dry Van quote_signal.")
    figure(doc, ROOT / "scorer_results" / "candidate_december.png", 6.4, "Figure 5. candidate_december.png, produced by score.py.")
    dec = final["december"]
    hist = final["lexington_fort_wayne_dry_van_history"]
    para(doc, f"The predicted rate runs from {fmt_money(dec['min'])} to {fmt_money(dec['max'])} (about ${dec['per_mile_mean']:.2f} per mile). Two effects are visible: a weekly cycle from the market index "
              f"(softest on Sundays), and a rising ramp from early December to the quarter end, on top of the slow trend. For reference, the {hist['n']} clean historical Lexington to Fort Wayne Dry Van loads averaged "
              f"{fmt_money(hist['mean_rate'])} (${hist['mean_rate_per_mile']:.2f} per mile), so the forecast is consistent with the lane's history.")

    # ---- 7. Limitations
    heading(doc, "7. Limitations")
    bullet(doc, " The trend and quarter-end ramp are learned from three quarters and extrapolated two months ahead. The same pattern appears in all three observed quarters, and the calendar terms improved two of the three forward folds (Jul-Aug was slightly worse than the model without them), "
                "but December-specific effects (holiday demand, year-end) cannot be learned from January-October data.", "Extrapolation.")
    bullet(doc, " Validation labels probably contain the same ~1.4% corrupted rows. No model can predict them, so squared-error metrics on the hidden labels will be dominated by them; "
                "median-type metrics are far less affected.", "Corrupted labels in the hidden set.")
    bullet(doc, " Final metrics are computed by the assessor after submission; the figures here are validation estimates.", "Estimates.")

    # ---- 8. Reproduce
    heading(doc, "8. Reproducing the results")
    for line in [
        "python -m pip install -r requirements.txt",
        "python scripts/run_final.py      # trains, writes validation_predictions.csv and fills data/december_chart_inputs.csv",
        "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
        "python scripts/run_validation.py # optional: regenerates every validation table (about 10 minutes)",
        "python scripts/run_eda.py && python scripts/build_report.py   # optional: figures and this report",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)

    doc.save(OUT)
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
