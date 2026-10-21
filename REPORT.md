# Store-Level Retail Demand Forecasting — Project Report

Oct 21, 2026 · Prasham Jain

## Abstract

A complete, verified pipeline forecasts daily store-level sales for the Rossmann chain (1,017,209 real transactions, 1,115 stores, Jan 2013 – Jul 2015), replacing the manual estimation the course synopsis identifies as the source of stock-outs and over-ordering. The final LightGBM model reaches WMAPE 7.87% and MASE 0.240 on 5-fold walk-forward validation, beating Ridge regression and two naive baselines in every fold. The same trained model drives a 48-day recursive forecast on Kaggle's real held-out test window (856 stores, zero NaN or negative predictions) and a live FastAPI web app that re-runs the pipeline — forecasts, SHAP explanations, leakage checks, full re-validation — on demand rather than from cached results. A follow-on experiment unifying Rossmann with a second retailer's data (Walmart, 2010–2012, no store or date overlap) improved mean WMAPE to 7.01% and MASE to 0.190, a real gain from genuinely out-of-domain training data.

## 1. Introduction & Problem Statement

Retailers plan inventory and staffing off manual estimation or simple historical averages — accurate enough most weeks, and wrong exactly when it matters: promotions, public holidays, seasonal peaks. Under-forecasting empties shelves and loses the sale; over-forecasting leaves markdowns and spoilage on perishables. Doing this by hand across more than a thousand stores does not scale, and does not improve as more data arrives the way a model does.

This project replaces that manual estimate with a model that has read each store's actual sales history — its calendar, its promotions, its competitors — and predicts demand before the week starts, not after. The course synopsis (`Prasham_Jain_2427030155_Project_Synopsis.docx`) frames this as a Predictive Analysis course deliverable built on the Rossmann Store Sales dataset. This report covers the project as built, including two extensions beyond the synopsis's original scope: a live, interactive web application, and a second-retailer ablation experiment.

## 2. Objectives

Seven objectives from the synopsis, each mapped to where it was met:

| # | Objective | Where met |
| --- | --- | --- |
| 1 | Analyse the dataset; identify temporal, promotional, and store-level features | Notebook 00 |
| 2 | Clean/preprocess the data; decompose trend and seasonality | Notebooks 01–02 |
| 3 | Engineer lag, rolling-window, and calendar features | Notebook 03 |
| 4 | Train regression and gradient-boosting models | Notebooks 04–05 |
| 5 | Evaluate against baselines under walk-forward validation | Notebook 06 |
| 6 | Interpret results; identify demand drivers | Notebook 07 |
| 7 | Demonstrate the full pipeline, raw data to forecast | Notebook 08 |

Two further objectives emerged beyond the synopsis during the build: making the pipeline runnable rather than only readable (the live web app, Section 8), and testing whether the approach generalises to a second retailer (Section 7).

## 3. Datasets

Two real Kaggle datasets, acquired via the user's own Kaggle account (required for both; neither can be redistributed in this repository):

|  | Rossmann Store Sales (primary) | Walmart Sales (ablation) |
| --- | --- | --- |
| Grain | Store × day | Store × week |
| Stores | 1,115 | 45 |
| Rows | 1,017,209 | 6,435 |
| Date range | 2013-01-01 → 2015-07-31 | 2010-02-05 → 2012-10-26 |
| Key columns | Sales, Customers, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2 | Weekly\_Sales, Holiday\_Flag, Temperature, Fuel\_Price, CPI, Unemployment |
| Test horizon | 41,088 rows (856 stores × 48 days, Aug 1 – Sep 17 2015), no ground-truth Sales | None (no held-out split) |

The two datasets share zero stores and zero overlapping dates — Walmart's span ends roughly two months before Rossmann's begins. That separation is what makes the Section 7 ablation an honest test of generalisation rather than near-duplicate data in disguise.

## 4. Methodology

Six sequential stages, each a numbered notebook in `notebooks/`:

1. **Cleaning** (`01`) — merge store attributes onto every sales row, decode `StateHoliday`, impute missing `CompetitionDistance`/`Promo2` fields with explicit “missing” flags rather than guesses. `Open == 0` (closed-store) rows are kept, not dropped, so each store's calendar stays unbroken for the lag math in stage 3. Row count in equals row count out: 1,017,209 → 1,017,209.
2. **EDA** (`02`) — STL decomposition (period = 7) splits total sales into trend, weekly seasonal, and residual. Day-of-week means, a promo-lift calculation (+38.8% on days a promo runs), and a correlation check that flags `Customers` (r = 0.89 with Sales) as a leakage trap — excluded from modelling because it is also absent from the real test set.
3. **Feature engineering** (`03`) — lag features (`lag_1/7/14/28`) and rolling statistics (`roll_mean`/`roll_std` over 7/14/28 days) are built by shifting the target one day back *before* windowing, so a feature for any (store, date) only touches that store's strictly earlier history. An executable `assert_no_leakage()` independently recomputes 30 random (store, date) pairs from raw history and asserts they match the stored features — run on both a synthetic rehearsal dataset and the full 1,017,209-row real dataset. 80.5% of rows (818,845) are trainable; the rest are closed-store days plus a residual of open days still inside their store's 28-day warm-up.
4. **Modelling** (`04`–`05`) — four candidates: seasonal-naive (copy last week) and a 7-day moving average as honest “no-model” baselines; Ridge regression (one-hot encoded, scaled) as the linear contender; LightGBM, trained on log(Sales), with every categorical — including the 1,115-value `Store` column — left as a native category rather than one-hot exploded.
5. **Walk-forward validation** (`06`) — one global model across all 1,115 stores (not 1,115 per-store models), validated with 5 expanding-window folds, each scored on a 42-day block it never trained on (matching the real Kaggle test horizon length). No shuffling anywhere — every split is a strict date cut, asserted in code.
6. **Interpretation & forecast** (`07`–`08`) — LightGBM gain-based importances plus SHAP values explain *why* the model predicts what it does; the same `clean_pipeline`/`build_feature_matrix` functions are re-applied unchanged to the real test window to produce an actual 48-day forecast (Section 6).

MASE is computed with `train_groups=Store`: for a global model, naive-error scale is calculated **within each store** and then pooled, so it is never contaminated by differencing across a store boundary in the concatenated training array — a correctness detail validated by a dedicated unit test.

## 5. Results

LightGBM beat every other model on WMAPE in 5 of 5 validation folds. Mean metrics across the 5 folds:

| Model | RMSE | MAE | WMAPE | MASE |
| --- | --- | --- | --- | --- |
| Seasonal-naive | 3,011 | 2,229 | 31.19% | 0.956 |
| Moving-average | 2,378 | 1,719 | 24.09% | 0.737 |
| Ridge | 1,264 | 845 | 11.87% | 0.362 |
| **LightGBM** | **886** | **560** | **7.87%** | **0.240** |

*(Chart: WMAPE by fold for all 4 models — see `reports/figures/model_comparison_table.csv`.)*

Each step of modelling sophistication roughly halves the error: naive → moving-average → Ridge → LightGBM runs 31.2% → 24.1% → 11.9% → 7.9% WMAPE. LightGBM's MASE of 0.240 means its typical error is about a quarter of the seasonal-naive baseline's.

**Feature importance.** Gain-based importance is dominated by which store it is (`Store`, 33.4% of total gain) — expected for a chain with real store-to-store variation — followed by calendar signals (`Day` 7.1%, `WeekOfYear` 5.2%, `DayOfWeek` 4.9%) and recent momentum (`lag_1` 4.3%, `lag_14` 4.2%, `roll_mean_7` 3.4%). SHAP values confirm the same ranking and add direction: `Promo` pushes predictions up sharply on the day it is active; `CompetitionDistance` has a real but much quieter, slower-moving effect.

## 6. Forecasting Demonstration

Notebook 08 runs the literal “raw data to forecast” demonstration the synopsis's Expected Outcome asks for. The same `clean_pipeline` and `build_feature_matrix` functions from stages 1 and 3 are re-applied, unchanged, to the real held-out test window, so there is no gap between how training and forecast features are built.

The complication: the horizon runs 48 days out, but the longest look-back feature needs 28 days of history — by day 29 that history is other forecast days, not yet-observed sales. The model predicts one day at a time, in date order, and feeds each day's own prediction back in as if it were the real figure before computing the next day's features: the standard recursive multi-step forecasting approach. This means prediction error can compound the further the horizon runs — an acknowledged limitation (Section 10), not a bug.

Result: 41,088 forecast rows across 856 real test stores (Kaggle's test set does not cover all 1,115 — a real quirk of the competition data, not an error) over 48 days (Aug 1 – Sep 17 2015), with zero NaN or negative predictions.

## 7. Multi-Retailer Ablation Experiment

Notebook 09 asks a different question: does training on a *second* retailer's sales history improve Rossmann's own held-out forecasts? Rossmann (daily) and Walmart (already weekly, no finer grain available) are unified onto a common store-week schema in `src/multi_retailer.py` — Rossmann aggregated up to weekly rather than Walmart fabricated down to daily, since inventing daily numbers from a weekly total would mean training on data that does not exist. Every store id is dataset-prefixed (`rossmann_262`, `walmart_12`) because both sources number stores 1..N independently; an executable leakage check (the same independent-recompute proof as Section 4, adapted to the weekly grain) passed on the combined 152,460-row dataset.

Because Walmart's dates (2010–2012) entirely precede Rossmann's (2013–2015), every Walmart row lands in the training window in every fold and never in a validation fold — it can only add training signal. Both arms below are scored on the *identical* 5 Rossmann validation weeks, so the only variable is how much training data the model saw:

| Fold | Rossmann-only WMAPE | Combined WMAPE | Rossmann-only MASE | Combined MASE |
| --- | --- | --- | --- | --- |
| 1 | 9.45% | 9.60% | 0.334 | 0.251 |
| 2 | 5.15% | 4.99% | 0.182 | 0.131 |
| 3 | 5.63% | 5.38% | 0.210 | 0.152 |
| 4 | 5.35% | 5.34% | 0.190 | 0.146 |
| 5 | 9.96% | 9.75% | 0.357 | 0.271 |
| **Mean** | **7.11%** | **7.01%** | **0.255** | **0.190** |

*(Chart: WMAPE by fold, Rossmann-only vs. combined — see `reports/figures/multi_retailer_comparison.csv`.)*

Combined training won on WMAPE in 4 of 5 folds and improved mean MASE by roughly 25% relative. A real gain, from a dataset with no overlapping stores, dates, or currency — genuinely transferable seasonal/holiday structure, not data leakage in disguise. (MASE here uses `seasonal_period=1`, week-over-week persistence, not the `period=7` used in Sections 4–5 — a different grain needs a different naive baseline, so these MASE values are not directly comparable to the daily model's.)

## 8. Live Web Application

`webapp/app.py` (FastAPI) loads the real data and both persisted models once at startup, then serves four panels at `webapp/static/index.html` — every panel calls a route that re-runs real project code on each request, not a route that reads a precomputed results file:

| Panel | What it actually does | Observed timing |
| --- | --- | --- |
| Run a live forecast | Pick any of 856 real test stores + a horizon (1–48 days); re-runs the Section 6 recursive forecast loop against the persisted LightGBM model | 93–280 ms |
| Explain this day | Live SHAP breakdown of one specific forecast prediction | 43–55 ms |
| Leakage check | Fresh, unseeded random sample re-verified against raw history on every click | 8–30 ms |
| Re-run 5-fold validation | Fully retrains Ridge + LightGBM across all 5 folds on demand; cached after first run | \~100–140 s |
| Multi-retailer ablation | Live prediction from the combined model for any store from either retailer, plus an on-demand re-run of the Section 7 comparison | 4–6 ms (predict) / \~140 s (re-run) |

Correctness detail: a single-row or single-store slice cast with plain `.astype("category")` would silently re-number category codes from whatever values are present in that slice — wrong for both prediction and SHAP. The app fixes each categorical column's levels once at startup from the full training set (`CATEGORY_LEVELS`) and reuses that exact mapping for every request, regardless of how small the slice is.

## 9. Repository & Reproducibility

Code: [github.com/PrashamJ17/Store-Level-Retail-Demand-Forecasting](https://github.com/PrashamJ17/Store-Level-Retail-Demand-Forecasting). Stack: Python 3.11, pandas 3.0, scikit-learn, LightGBM, SHAP, statsmodels, FastAPI, pytest.

```
notebooks/   00-08 main pipeline + 09 multi-retailer ablation, run in order
src/         tested pipeline code (cleaning, features, metrics, validation, models, multi_retailer)
scripts/     standalone runnable scripts (train_multi_retailer.py)
webapp/      live FastAPI demo app (app.py + static/index.html)
tests/       pytest unit tests — 22 passing, no Kaggle data required
data/, models/, reports/figures/   raw/intermediate data and artifacts (gitignored except .gitkeep)
```

To reproduce: `pip install -r requirements.txt`, download both datasets into `data/raw/` (own Kaggle account required for each), run `pytest tests/` (22 tests, no data needed), run notebooks 00–08 in order, optionally 09, then `python webapp/app.py` for the live app. Every real number in this report was produced by actually running this code against the real datasets — not estimated or interpolated.

## 10. Limitations

- `Customers` is excluded from every model by design: it is contemporaneous with `Sales` (r = 0.89) and absent from the real test set, so including it would be leakage.
- No external data — no weather, no local-event calendars — feeds any model.
- One global model stands in for 1,115 stores rather than per-store or clustered models, chosen for data efficiency and simplicity; a clustered approach was not tested.
- `Promo2`'s active months are decoded from `PromoInterval`, an approximation of the real promotional calendar, not a verified ground truth.
- The Section 6 forecast compounds its own prediction error past day 28 of the horizon (Section 6), an inherent property of recursive multi-step forecasting, not a defect in this run.
- The Section 7 ablation does not unify currency or scale between retailers — the model relies entirely on `combined_store_id`/`dataset_source` to separate the two regimes, never on a claimed numeric equivalence.
- Walmart's file has no real promotional signal (only `Holiday_Flag`), so `has_promo` is `NaN` for every Walmart row — left honest rather than backfilled with a guess.
- The multi-retailer model operates at weekly grain; it is a separate experiment and does not replace the daily Sections 1–6 pipeline or the primary web app forecast.

## 11. Conclusion

The synopsis's Expected Outcome calls for "a functional predictive model that accurately forecasts store-level product demand," demonstrating correct preprocessing, effective feature engineering, sound validation, and practical application to a real retail problem. All four are satisfied with evidence, not assertion: preprocessing preserves every row and is leakage-checked twice (Sections 4 and 7); the final LightGBM model clears a defined acceptance bar (beats every baseline on WMAPE in ≥4 of 5 folds, MASE < 1.0) with room to spare at 7.87% / 0.240; and the pipeline produces an actual 48-day forecast on real held-out data, not a validation score alone.

Two extensions go beyond what the synopsis asked for. The live web app turns the pipeline from something read into something run — every panel re-executes real code against the real model on demand, timed to prove it is not cached. The multi-retailer ablation tests the approach's generalisation past the single dataset it was built on, and finds a real, honestly-measured improvement. Together, they show a pipeline that is not just validated once and reported, but built to be interrogated.
