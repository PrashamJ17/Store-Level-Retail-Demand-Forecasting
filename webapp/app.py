"""Live demo backend for the Store-Level Retail Demand Forecasting pipeline.

Every /api route in this file genuinely re-runs real project code against the
real Rossmann dataset and the real persisted model -- nothing here reads from
a precomputed results file. Run with `python webapp/app.py` from anywhere;
it adds the project root to sys.path itself.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.cleaning import clean_pipeline
from src.data_io import read_raw_csvs
from src.features import build_feature_matrix, filter_trainable_rows
from src.metrics import evaluate_all
from src.models import build_lightgbm_model, build_ridge_pipeline, moving_average_predict, seasonal_naive_predict
from src.multi_retailer import (
    add_weekly_features,
    aggregate_to_weekly,
    filter_trainable_weekly,
    prep_rossmann,
    prep_walmart,
)
from src.validation import walk_forward_splits

HISTORY_DAYS = 60
MAX_HORIZON = 48

print("[startup] loading raw data and the persisted model...")
_t0 = time.time()

train_raw, test_raw, store_raw = read_raw_csvs(PROJECT_ROOT / "data" / "raw")
train_clean = clean_pipeline(train_raw, store_raw)
test_clean = clean_pipeline(test_raw, store_raw)
test_clean["Sales"] = np.nan
test_clean["Customers"] = np.nan

full_engineered = build_feature_matrix(train_clean)
trainable_df = filter_trainable_rows(full_engineered)

model = joblib.load(PROJECT_ROOT / "models" / "lightgbm_global.pkl")
config = joblib.load(PROJECT_ROOT / "models" / "feature_config.pkl")
FEATURE_COLS = config["feature_cols"]
NUMERIC_FEATURE_COLS = config["numeric_cols"]
CATEGORICAL_FEATURE_COLS = config["categorical_cols"]

# Fix the exact categorical levels the persisted model was trained on, so a
# single-row (or single-store) slice at request time gets cast to the SAME
# category codes as training -- casting a tiny subset with plain
# `.astype("category")` would silently re-number categories from whatever's
# present in that subset, which is wrong for both prediction and SHAP.
_train_cat_reference = trainable_df.copy()
for _col in CATEGORICAL_FEATURE_COLS:
    _train_cat_reference[_col] = _train_cat_reference[_col].astype("category")
CATEGORY_LEVELS = {col: _train_cat_reference[col].cat.categories for col in CATEGORICAL_FEATURE_COLS}


def cast_fixed_categoricals(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURE_COLS:
        df[col] = pd.Categorical(df[col], categories=CATEGORY_LEVELS[col])
    return df


explainer = shap.TreeExplainer(model)

_history_all = train_clean.sort_values(["Store", "Date"]).groupby("Store", group_keys=False).tail(HISTORY_DAYS)
history_by_store = {
    store_id: g.reset_index(drop=True)
    for store_id, g in _history_all.groupby("Store")
}
test_by_store = {
    store_id: g.sort_values("Date").reset_index(drop=True)
    for store_id, g in test_clean.groupby("Store")
}

_store_meta_df = (
    store_raw[store_raw["Store"].isin(test_by_store.keys())]
    [["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]]
    .sort_values("Store")
)
STORE_META = _store_meta_df.astype(object).where(pd.notna(_store_meta_df), None).to_dict(orient="records")

print(f"[startup] ready in {time.time() - _t0:.1f}s — {len(train_clean):,} train rows, "
      f"{len(test_by_store)} test stores, model expects {len(FEATURE_COLS)} features")

# --- multi-retailer (notebook 09) artifacts, loaded alongside the daily model ---
_mr_path = PROJECT_ROOT / "models" / "lightgbm_multiretailer_weekly.pkl"
MR_AVAILABLE = _mr_path.exists() and (PROJECT_ROOT / "data" / "raw" / "Walmart_Sales.csv").exists()
mr_model = mr_config = mr_trainable = mr_category_levels = mr_rossmann_only = None
MR_FEATURE_COLS = []

if MR_AVAILABLE:
    print("[startup] loading Walmart data and the multi-retailer model...")
    _mr_t0 = time.time()
    walmart_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Walmart_Sales.csv")
    _combined_prepped = pd.concat([prep_rossmann(train_clean), prep_walmart(walmart_raw)], ignore_index=True)
    _mr_weekly = aggregate_to_weekly(_combined_prepped)
    _mr_full = add_weekly_features(_mr_weekly)
    mr_trainable = filter_trainable_weekly(_mr_full)
    mr_rossmann_only = mr_trainable[mr_trainable["dataset_source"] == "rossmann"].reset_index(drop=True)

    mr_model = joblib.load(_mr_path)
    mr_config = joblib.load(PROJECT_ROOT / "models" / "multiretailer_feature_config.pkl")
    MR_FEATURE_COLS = mr_config["feature_cols"]
    mr_category_levels = mr_config["category_levels"]
    print(f"[startup] multi-retailer ready in {time.time() - _mr_t0:.1f}s — "
          f"{len(mr_trainable):,} store-weeks across {mr_trainable['dataset_source'].nunique()} sources")
else:
    print("[startup] multi-retailer model/data not found — /api/multi-retailer/* routes will 404. "
          "Run `python scripts/train_multi_retailer.py` first.")


def mr_cast_categoricals(df):
    df = df.copy()
    for col in mr_config["categorical_cols"]:
        df[col] = pd.Categorical(df[col], categories=mr_category_levels[col])
    return df


_validate_cache = {"result": None, "computed_at": None, "elapsed_s": None}
_mr_validate_cache = {"result": None}

app = FastAPI(title="Forecasting Ledger — Live Demo")


def _recursive_forecast(store_id, horizon):
    if store_id not in test_by_store:
        raise HTTPException(status_code=404, detail=f"Store {store_id} is not in the real Kaggle test set.")
    test_rows_all = test_by_store[store_id]
    horizon = max(1, min(horizon, len(test_rows_all), MAX_HORIZON))
    test_rows = test_rows_all.head(horizon)

    running_history = history_by_store[store_id].copy()
    predictions = []
    feature_rows = {}

    for _, today_row in test_rows.iterrows():
        forecast_date = today_row["Date"]
        today_df = test_rows_all[test_rows_all["Date"] == forecast_date].copy()

        combined = pd.concat([running_history, today_df], ignore_index=True)
        engineered = build_feature_matrix(combined)
        today_engineered = engineered[engineered["Date"] == forecast_date].copy()

        today_cat = cast_fixed_categoricals(today_engineered)
        feature_row = today_cat[FEATURE_COLS].iloc[[0]]
        pred_log = float(model.predict(feature_row)[0])
        pred_sales = float(np.expm1(pred_log))
        is_open = int(today_engineered["Open"].iloc[0])
        if not is_open:
            pred_sales = 0.0

        date_str = forecast_date.strftime("%Y-%m-%d")
        predictions.append({
            "date": date_str,
            "predicted_sales": round(pred_sales, 1),
            "open": is_open,
            "promo": int(today_engineered["Promo"].iloc[0]),
            "state_holiday": str(today_engineered["state_holiday_type"].iloc[0]),
            "school_holiday": int(today_engineered["SchoolHoliday"].iloc[0]),
        })
        feature_rows[date_str] = feature_row

        today_with_pred = today_df.copy()
        today_with_pred["Sales"] = pred_sales
        running_history = pd.concat([running_history, today_with_pred], ignore_index=True).tail(HISTORY_DAYS)

    return predictions, feature_rows


@app.get("/api/meta")
def api_meta():
    return {
        "n_rows": len(train_clean),
        "n_stores": int(train_clean["Store"].nunique()),
        "date_min": train_clean["Date"].min().strftime("%Y-%m-%d"),
        "date_max": train_clean["Date"].max().strftime("%Y-%m-%d"),
        "trainable_rows": len(trainable_df),
        "trainable_pct": round(len(trainable_df) / len(full_engineered) * 100, 1),
        "test_stores": len(test_by_store),
        "test_days": int(test_clean["Date"].nunique()),
        "feature_count": len(FEATURE_COLS),
        "model_type": "LightGBM (global, log-target)",
    }


@app.get("/api/stores")
def api_stores():
    return STORE_META


@app.get("/api/forecast")
def api_forecast(store: int, horizon: int = 28):
    t0 = time.time()
    predictions, _ = _recursive_forecast(store, horizon)

    hist = history_by_store[store][["Date", "Sales"]].copy()
    hist_out = [{"date": d.strftime("%Y-%m-%d"), "sales": float(s)} for d, s in zip(hist["Date"], hist["Sales"])]

    meta_row = next((m for m in STORE_META if m["Store"] == store), None)

    return JSONResponse({
        "store": store,
        "store_meta": meta_row,
        "history": hist_out,
        "forecast": predictions,
        "computed_in_ms": round((time.time() - t0) * 1000, 1),
    })


@app.get("/api/explain")
def api_explain(store: int, date: str):
    t0 = time.time()
    test_rows_all = test_by_store.get(store)
    if test_rows_all is None:
        raise HTTPException(status_code=404, detail=f"Store {store} is not in the real Kaggle test set.")
    target_dates = test_rows_all["Date"]
    try:
        target_idx = list(target_dates.dt.strftime("%Y-%m-%d")).index(date)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"{date} is not in store {store}'s test window.")

    predictions, feature_rows = _recursive_forecast(store, target_idx + 1)
    row = feature_rows[date]
    pred_entry = predictions[-1]

    shap_values = explainer.shap_values(row)
    base_value = float(explainer.expected_value)
    contributions = []
    for col, val, sv in zip(FEATURE_COLS, row.iloc[0].tolist(), shap_values[0]):
        contributions.append({
            "feature": col,
            "value": str(val),
            "shap_value": float(sv),
        })
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return JSONResponse({
        "store": store,
        "date": date,
        "predicted_sales": pred_entry["predicted_sales"],
        "base_value_log": base_value,
        "top_contributions": contributions[:10],
        "computed_in_ms": round((time.time() - t0) * 1000, 1),
    })


@app.get("/api/leakage-check")
def api_leakage_check(sample_size: int = 8):
    t0 = time.time()
    rng = np.random.default_rng()  # unseeded -- a fresh sample every call
    candidates = full_engineered.index[full_engineered["lag_7"].notna() & full_engineered["roll_mean_7"].notna()].to_numpy()
    sample_idx = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)

    rows = []
    all_match = True
    for idx in sample_idx:
        row = full_engineered.loc[idx]
        store_id, date = row["Store"], row["Date"]
        history = full_engineered[(full_engineered["Store"] == store_id) & (full_engineered["Date"] < date)].sort_values("Date")
        if len(history) < 7:
            continue
        expected_lag_7 = float(history["Sales"].iloc[-7])
        expected_roll_mean_7 = float(history["Sales"].iloc[-7:].mean())
        stored_lag_7 = float(row["lag_7"])
        stored_roll_mean_7 = float(row["roll_mean_7"])
        match = bool(np.isclose(stored_lag_7, expected_lag_7) and np.isclose(stored_roll_mean_7, expected_roll_mean_7))
        all_match = all_match and match
        rows.append({
            "store": int(store_id),
            "date": date.strftime("%Y-%m-%d"),
            "stored_lag_7": round(stored_lag_7, 1),
            "recomputed_lag_7": round(expected_lag_7, 1),
            "stored_roll_mean_7": round(stored_roll_mean_7, 1),
            "recomputed_roll_mean_7": round(expected_roll_mean_7, 1),
            "match": match,
        })

    return JSONResponse({
        "passed": all_match,
        "samples": rows,
        "computed_in_ms": round((time.time() - t0) * 1000, 1),
    })


@app.post("/api/validate")
def api_validate(force: bool = False):
    if _validate_cache["result"] is not None and not force:
        return JSONResponse({**_validate_cache["result"], "cached": True})

    t0 = time.time()
    df = trainable_df.sort_values(["Store", "Date"]).reset_index(drop=True)
    splits = walk_forward_splits(df["Date"], n_folds=5, val_days=42)

    results = []
    for split in splits:
        train_df = df.loc[split["train_mask"]]
        val_df = df.loc[split["val_mask"]]

        row = {"fold": split["fold"]}
        for name, pred_fn in [("seasonal_naive", seasonal_naive_predict), ("moving_average", moving_average_predict)]:
            pred = pred_fn(val_df)
            m = evaluate_all(val_df["Sales"], pred, train_df["Sales"], seasonal_period=7, train_groups=train_df["Store"])
            row.update({f"{name}_{k}": v for k, v in m.items()})

        ridge = build_ridge_pipeline(NUMERIC_FEATURE_COLS, CATEGORICAL_FEATURE_COLS)
        ridge.fit(train_df[FEATURE_COLS], train_df["log_sales"])
        ridge_pred = np.expm1(ridge.predict(val_df[FEATURE_COLS]))
        m = evaluate_all(val_df["Sales"], ridge_pred, train_df["Sales"], seasonal_period=7, train_groups=train_df["Store"])
        row.update({f"ridge_{k}": v for k, v in m.items()})

        train_cat = cast_fixed_categoricals(train_df)
        val_cat = cast_fixed_categoricals(val_df)
        lgbm = build_lightgbm_model()
        lgbm.fit(train_cat[FEATURE_COLS], train_cat["log_sales"])
        lgbm_pred = np.expm1(lgbm.predict(val_cat[FEATURE_COLS]))
        m = evaluate_all(val_df["Sales"], lgbm_pred, train_df["Sales"], seasonal_period=7, train_groups=train_df["Store"])
        row.update({f"lightgbm_{k}": v for k, v in m.items()})

        results.append(row)

    results_df = pd.DataFrame(results)
    model_names = ["seasonal_naive", "moving_average", "ridge", "lightgbm"]
    avg_metrics = {m: {k: float(results_df[f"{m}_{k}"].mean()) for k in ["rmse", "mae", "wmape", "mase"]} for m in model_names}
    wins = int((results_df["lightgbm_wmape"] == results_df[[f"{m}_wmape" for m in model_names]].min(axis=1)).sum())

    elapsed = time.time() - t0
    payload = {
        "n_folds": len(results_df),
        "model_avg_metrics": avg_metrics,
        "lightgbm_fold_wins": wins,
        "elapsed_s": round(elapsed, 1),
    }
    _validate_cache["result"] = payload
    return JSONResponse({**payload, "cached": False})


def _require_mr():
    if not MR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Multi-retailer model/data not found. Run `python scripts/train_multi_retailer.py` first.",
        )


@app.get("/api/multi-retailer/meta")
def api_mr_meta():
    _require_mr()
    by_source = mr_trainable["dataset_source"].value_counts().to_dict()
    return {
        "trainable_store_weeks": len(mr_trainable),
        "stores_by_source": {
            src: int(mr_trainable.loc[mr_trainable["dataset_source"] == src, "combined_store_id"].nunique())
            for src in by_source
        },
        "rows_by_source": {k: int(v) for k, v in by_source.items()},
        "date_min": mr_trainable["week_start"].min().strftime("%Y-%m-%d"),
        "date_max": mr_trainable["week_start"].max().strftime("%Y-%m-%d"),
        "feature_count": len(MR_FEATURE_COLS),
    }


@app.get("/api/multi-retailer/stores")
def api_mr_stores():
    _require_mr()
    grouped = mr_trainable.groupby("combined_store_id").agg(
        dataset_source=("dataset_source", "first"),
        week_min=("week_start", "min"),
        week_max=("week_start", "max"),
        n_weeks=("week_start", "count"),
    )
    return [
        {
            "combined_store_id": store_id,
            "dataset_source": row["dataset_source"],
            "week_min": row["week_min"].strftime("%Y-%m-%d"),
            "week_max": row["week_max"].strftime("%Y-%m-%d"),
            "n_weeks": int(row["n_weeks"]),
        }
        for store_id, row in grouped.sort_index().iterrows()
    ]


@app.get("/api/multi-retailer/predict")
def api_mr_predict(store: str, week: str):
    _require_mr()
    t0 = time.time()
    week_ts = pd.Timestamp(week)
    match = mr_trainable[(mr_trainable["combined_store_id"] == store) & (mr_trainable["week_start"] == week_ts)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No trainable row for {store} at week {week}.")

    row = match.iloc[[0]]
    row_cat = mr_cast_categoricals(row)
    pred_log = float(mr_model.predict(row_cat[MR_FEATURE_COLS])[0])
    predicted = float(np.expm1(pred_log))
    actual = float(row["weekly_sales"].iloc[0])

    return JSONResponse({
        "store": store,
        "week": week,
        "dataset_source": row["dataset_source"].iloc[0],
        "predicted_sales": round(predicted, 1),
        "actual_sales": round(actual, 1),
        "error_pct": round(abs(predicted - actual) / actual * 100, 2) if actual else None,
        "has_promo": bool(row["has_promo"].iloc[0]) if pd.notna(row["has_promo"].iloc[0]) else None,
        "has_holiday": bool(row["has_holiday"].iloc[0]),
        "computed_in_ms": round((time.time() - t0) * 1000, 1),
    })


def _mr_run_walk_forward(df):
    splits = walk_forward_splits(df["week_start"], n_folds=5, val_days=42)
    results = []
    for split in splits:
        train_df = df.loc[split["train_mask"]]
        val_df = df.loc[split["val_mask"]]

        train_cat = mr_cast_categoricals(train_df)
        val_cat = mr_cast_categoricals(val_df)
        model_ = build_lightgbm_model()
        model_.fit(train_cat[MR_FEATURE_COLS], train_cat["log_weekly_sales"])
        pred = np.expm1(model_.predict(val_cat[MR_FEATURE_COLS]))

        m = evaluate_all(val_df["weekly_sales"], pred, train_df["weekly_sales"],
                          seasonal_period=1, train_groups=train_df["combined_store_id"])
        m["fold"] = split["fold"]
        results.append(m)
    return pd.DataFrame(results)


@app.post("/api/multi-retailer/validate")
def api_mr_validate(force: bool = False):
    _require_mr()
    if _mr_validate_cache["result"] is not None and not force:
        return JSONResponse({**_mr_validate_cache["result"], "cached": True})

    t0 = time.time()
    results_a = _mr_run_walk_forward(mr_rossmann_only)
    results_b = _mr_run_walk_forward(mr_trainable)

    per_fold = [
        {
            "fold": int(a_row["fold"]),
            "rossmann_only_wmape": float(a_row["wmape"]),
            "combined_wmape": float(b_row["wmape"]),
            "rossmann_only_mase": float(a_row["mase"]),
            "combined_mase": float(b_row["mase"]),
        }
        for (_, a_row), (_, b_row) in zip(results_a.iterrows(), results_b.iterrows())
    ]
    mean_a_wmape, mean_b_wmape = float(results_a["wmape"].mean()), float(results_b["wmape"].mean())
    folds_combined_wins = int((results_b["wmape"].values < results_a["wmape"].values).sum())

    payload = {
        "per_fold": per_fold,
        "mean_rossmann_only_wmape": mean_a_wmape,
        "mean_combined_wmape": mean_b_wmape,
        "mean_rossmann_only_mase": float(results_a["mase"].mean()),
        "mean_combined_mase": float(results_b["mase"].mean()),
        "folds_combined_wins": folds_combined_wins,
        "n_folds": len(per_fold),
        "elapsed_s": round(time.time() - t0, 1),
    }
    _mr_validate_cache["result"] = payload
    return JSONResponse({**payload, "cached": False})


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
