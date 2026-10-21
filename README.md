# Store-Level Retail Demand Forecasting

Predictive Analysis course project (Manipal University Jaipur) — a store-level
demand forecasting pipeline built on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
dataset. See `Prasham_Jain_2427030155_Project_Synopsis.docx` for the original
synopsis and objectives.

## Setup

```bash
# from the project root
~/.local/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pedictive --display-name "Python (pedictive)"
```

## Getting the data

The notebooks expect `data/raw/{train,test,store}.csv` from the Kaggle
`rossmann-store-sales` competition:

1. Accept the competition rules at kaggle.com (search "Rossmann Store Sales").
2. Create an API token: Account → Create New API Token → saves `kaggle.json`.
3. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
4. ```bash
   kaggle competitions download -c rossmann-store-sales -p data/raw/
   unzip data/raw/rossmann-store-sales.zip -d data/raw/
   ```

(Or download the zip manually from the competition's "Data" tab and extract
the three CSVs into `data/raw/`.)

## Running the pipeline

Run the notebooks in `notebooks/` **in numeric order** (00 → 08) — each one
loads a parquet artifact saved by the previous one:

| Notebook | Produces | Synopsis objective |
|---|---|---|
| `00_data_acquisition_and_overview` | `data/interim/train_raw_typed.parquet` | Obj 1 |
| `01_data_cleaning_and_preprocessing` | `data/interim/train_merged_clean.parquet` | Obj 2 (cleaning) |
| `02_exploratory_data_analysis` | figures in `reports/figures/` | Obj 2 (EDA) |
| `03_feature_engineering` | `data/processed/train_features.parquet` | Obj 3 |
| `04_baseline_and_linear_models` | — | Obj 4 (baseline + regression) |
| `05_gradient_boosting_model` | — | Obj 4 (boosting) |
| `06_walkforward_validation_and_evaluation` | `reports/figures/model_comparison_table.csv` | Obj 5 |
| `07_model_interpretation` | SHAP/importance figures | Obj 6 |
| `08_pipeline_summary_and_conclusions` | test-horizon forecast table | Obj 7 |

## Multi-retailer ablation (optional, additional)

`notebooks/09_multi_retailer_experiment.ipynb` asks a different question from
00–08: does training on a *second* retailer's sales history (Walmart, a
different currency, continent, and 2010–2012 date range with zero overlap
with Rossmann) improve Rossmann's own held-out forecasts? Requires
`data/raw/Walmart_Sales.csv` (the single-file "Walmart Sales" Kaggle
dataset — `Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price,
CPI, Unemployment`). Both datasets are unified onto a common store-week grain
via `src/multi_retailer.py`, with a dataset-prefixed `combined_store_id` to
prevent Rossmann/Walmart store-number collisions, and evaluated with the same
walk-forward scheme as notebook 06. This is a separate experiment with its
own model artifact (`models/lightgbm_multiretailer_weekly.pkl`) — it doesn't
change the daily pipeline or webapp above. Can also be run as a plain script:
`python scripts/train_multi_retailer.py`.

## Live demo app

```bash
python webapp/app.py
# then open http://127.0.0.1:8000
```

Requires `data/raw/{train,test,store}.csv` and a trained model (run notebook
`06_walkforward_validation_and_evaluation` once first — it persists
`models/lightgbm_global.pkl` and `models/feature_config.pkl`). Startup takes
10-30s while it loads and cleans the full dataset into memory.

This is not a slideshow of precomputed results — every panel in the browser
calls a live FastAPI endpoint that re-runs real project code on each request:

- **Run a live forecast** — pick any of the 856 real test-set stores and a
  horizon (1-48 days); it re-runs the exact recursive forecasting loop from
  notebook 08 against the persisted LightGBM model and returns a fresh
  prediction, timed in milliseconds.
- **Explain this day** — click any forecast day for a live SHAP breakdown of
  that specific prediction.
- **Leakage check** — re-runs the stage-03 leakage proof against a fresh,
  unseeded random sample on every click.
- **Re-run the 5-fold validation** — actually retrains Ridge and LightGBM
  across all 5 walk-forward folds on demand (~60-90s), the same evaluation as
  notebook 06, cached after the first run unless you force a re-run.

## Tests

```bash
pytest tests/
```

`tests/` covers `src/metrics.py`, `src/features.py`, and `src/multi_retailer.py`
against small synthetic series with hand-computed expected values — they
don't require the Kaggle dataset(s) and should pass immediately after
`pip install -r requirements.txt`.

## Project structure

```
data/{raw,interim,processed}/   # raw Kaggle CSVs, cleaned, model-ready (gitignored except .gitkeep)
notebooks/                      # 00-08 (main pipeline) + 09 (multi-retailer ablation), run in order
src/                            # shared, tested pipeline code imported by the notebooks
scripts/                        # standalone runnable scripts (e.g. train_multi_retailer.py)
models/                         # persisted model artifacts (gitignored)
reports/figures/                # exported charts for the write-up
tests/                          # pytest unit tests for src/metrics.py, src/features.py, src/multi_retailer.py
webapp/                         # live FastAPI demo app (app.py + static/index.html)
```
