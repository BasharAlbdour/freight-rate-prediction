# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions and `Freight_Rate_Report.docx` for the write-up (validation approach, data split, December chart).

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Quick start (reproduce the submission)

```bash
python -m pip install -r requirements.txt
python scripts/run_final.py        # ~10 s: trains the final model, writes validation_predictions.csv
                                   #        and fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`run_final.py` is deterministic (fixed seeds); predictions were verified bit-identical across pandas 2.x and 3.x.
Tested with Python 3.13, pandas 2.3, numpy 2.5, scikit-learn 1.9, matplotlib 3.11.

Optional, to regenerate every number and figure in the report (about 10 minutes in total):

```bash
python scripts/run_eda.py          # data-quality counts + figures        -> reports/
python scripts/run_validation.py   # model comparison, ablations, tuning  -> reports/
python scripts/run_final.py        # re-run so the saved model summary matches
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python scripts/build_report.py     # -> Freight_Rate_Report.docx
```

## Approach in one page

**Data quality.** 677 training labels (1.4%) are corrupted (0.17-0.45x or 2.2-5.4x what their features imply) and are removed from training only; weights have sign flips, gaps and a 47,500 lb cap; `market_index` has gaps; eight cities occur only in the validation file. Details in the report.

**Validation.** The 12,000 loads to predict (Nov-Dec 2025) come *after* all labeled data (Jan-Oct 2025), so validation is chronological: forward-chaining with two-month test folds, plus a hidden-city test. Random K-fold is computed only to show it is over-optimistic.

**Model.** `log(posted_rate)` = log-linear structure (ln distance, equipment, market index, linear time trend, quarter-end ramp) + histogram gradient boosting on the residuals (weight, per-load and same-day market signals, coordinates). Forward-chaining MAE on clean labels is about $38 per load (1.6%), versus $105 for a distance/equipment baseline.

## Repository layout

```
score.py                      provided scorer (unchanged)
requirements.txt              dependencies
validation_predictions.csv    submission file: load_id,predicted_rate
Freight_Rate_Report.docx      report (validation approach, split, December chart)
data/                         provided CSVs (december_chart_inputs.csv has predicted_rate filled in)
freight_rate/
    data.py                   loading, cleaning, label-outlier detection
    features.py               feature construction and feature groups
    models.py                 LinearPlusBoost model and the candidate list
    validation.py             forward-chaining, unseen-city and random CV
    chart.py                  features for the fixed December chart rows
scripts/
    run_final.py              train + write predictions
    run_validation.py         model comparison, tuning, ablations
    run_eda.py                data-quality summary and figures
    build_report.py           builds the DOCX from reports/ files
reports/                      result tables (CSV/JSON) and figures
scorer_results/               candidate_december.png from score.py
```

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link (talking points in `LOOM_SCRIPT.md`)
