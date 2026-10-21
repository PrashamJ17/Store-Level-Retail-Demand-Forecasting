"""Ablation experiment: does adding Walmart's weekly sales data to training
improve the model's forecasts on Rossmann's own held-out weeks?

Both setups are evaluated on the IDENTICAL 5-fold walk-forward validation,
anchored to Rossmann's date range (Walmart's dates all precede it, so
Walmart rows -- when included -- only ever add training signal; they never
appear in a validation fold). This keeps the comparison honest: the only
thing that changes between "Rossmann-only" and "combined" is how much
training data the model saw, not what it's scored against.

Run from the project root: python scripts/train_multi_retailer.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.cleaning import clean_pipeline
from src.data_io import read_raw_csvs
from src.metrics import evaluate_all
from src.models import build_lightgbm_model
from src.multi_retailer import (
    add_weekly_features,
    aggregate_to_weekly,
    assert_no_leakage_weekly,
    filter_trainable_weekly,
    prep_rossmann,
    prep_walmart,
)
from src.validation import walk_forward_splits

NUMERIC_FEATURE_COLS = [
    "lag_1", "lag_4", "lag_8", "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8",
    "has_promo", "has_holiday", "competition_distance", "store_size",
    "temperature", "fuel_price", "cpi", "unemployment",
    "year", "month", "week_of_year",
]
CATEGORICAL_FEATURE_COLS = ["combined_store_id", "dataset_source", "store_type_combined"]
FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


def cast_categoricals(df, category_levels):
    df = df.copy()
    for col in CATEGORICAL_FEATURE_COLS:
        df[col] = pd.Categorical(df[col], categories=category_levels[col])
    return df


def run_walk_forward(df, label, category_levels):
    """df: the TRAINING POOL (may include Walmart rows or not). Folds are
    computed from df's own date range, but since this function is always
    called with `full_trainable` restricted to rows <= the fold's val
    window, the fold boundaries are identical across both experiment arms
    (they're anchored on the max date, which is the same in both cases)."""
    splits = walk_forward_splits(df["week_start"], n_folds=5, val_days=42)
    results = []
    for split in splits:
        train_df = df.loc[split["train_mask"]]
        val_df = df.loc[split["val_mask"]]
        assert train_df["week_start"].max() < val_df["week_start"].min()

        train_cat = cast_categoricals(train_df, category_levels)
        val_cat = cast_categoricals(val_df, category_levels)

        model = build_lightgbm_model()
        model.fit(train_cat[FEATURE_COLS], train_cat["log_weekly_sales"])
        pred = np.expm1(model.predict(val_cat[FEATURE_COLS]))

        m = evaluate_all(
            val_df["weekly_sales"], pred, train_df["weekly_sales"],
            seasonal_period=1, train_groups=train_df["combined_store_id"],
        )
        m["fold"] = split["fold"]
        m["n_train"] = len(train_df)
        m["n_val"] = len(val_df)
        results.append(m)
        print(f"  [{label}] fold {split['fold']}: n_train={len(train_df):>7,} "
              f"wmape={m['wmape']*100:.2f}% mase={m['mase']:.3f}")
    return pd.DataFrame(results)


def main():
    t0 = time.time()
    print("Loading and cleaning Rossmann...")
    train_raw, test_raw, store_raw = read_raw_csvs(PROJECT_ROOT / "data" / "raw")
    train_clean = clean_pipeline(train_raw, store_raw)
    rossmann_prepped = prep_rossmann(train_clean)

    print("Loading Walmart...")
    walmart_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Walmart_Sales.csv")
    walmart_prepped = prep_walmart(walmart_raw)

    combined_prepped = pd.concat([rossmann_prepped, walmart_prepped], ignore_index=True)
    weekly = aggregate_to_weekly(combined_prepped)
    full = add_weekly_features(weekly)

    print("Leakage check on combined weekly data...")
    assert_no_leakage_weekly(full, sample_size=30)
    print("  PASSED")

    trainable = filter_trainable_weekly(full)
    print(f"Trainable store-weeks: {len(trainable):,} "
          f"({trainable['dataset_source'].value_counts().to_dict()})")

    category_levels = {col: pd.Categorical(trainable[col]).categories for col in CATEGORICAL_FEATURE_COLS}

    rossmann_only = trainable[trainable["dataset_source"] == "rossmann"].reset_index(drop=True)

    print("\n=== Arm A: Rossmann-only weekly ===")
    results_a = run_walk_forward(rossmann_only, "rossmann-only", category_levels)

    print("\n=== Arm B: Rossmann + Walmart combined ===")
    results_b = run_walk_forward(trainable, "combined", category_levels)

    summary = pd.DataFrame({
        "rossmann_only_wmape": results_a["wmape"],
        "combined_wmape": results_b["wmape"],
        "rossmann_only_mase": results_a["mase"],
        "combined_mase": results_b["mase"],
    })
    summary.index = results_a["fold"]
    print("\n=== Per-fold comparison ===")
    print(summary.to_string())

    mean_a_wmape, mean_b_wmape = results_a["wmape"].mean(), results_b["wmape"].mean()
    mean_a_mase, mean_b_mase = results_a["mase"].mean(), results_b["mase"].mean()
    folds_combined_wins = int((results_b["wmape"].values < results_a["wmape"].values).sum())

    print(f"\nMean WMAPE -- Rossmann-only: {mean_a_wmape*100:.3f}%  |  Combined: {mean_b_wmape*100:.3f}%")
    print(f"Mean MASE  -- Rossmann-only: {mean_a_mase:.4f}  |  Combined: {mean_b_mase:.4f}")
    print(f"Combined beat Rossmann-only on WMAPE in {folds_combined_wins}/5 folds")
    verdict = "Combined training HELPS" if mean_b_wmape < mean_a_wmape else "Combined training does NOT help"
    print(f"Verdict: {verdict} on this data.")

    print("\nFitting final combined model on all trainable rows...")
    final_cat = cast_categoricals(trainable, category_levels)
    final_model = build_lightgbm_model()
    final_model.fit(final_cat[FEATURE_COLS], final_cat["log_weekly_sales"])

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final_model, models_dir / "lightgbm_multiretailer_weekly.pkl")
    joblib.dump(
        {
            "numeric_cols": NUMERIC_FEATURE_COLS,
            "categorical_cols": CATEGORICAL_FEATURE_COLS,
            "feature_cols": FEATURE_COLS,
            "category_levels": category_levels,
        },
        models_dir / "multiretailer_feature_config.pkl",
    )

    reports_dir = PROJECT_ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [results_a.add_prefix("rossmann_only_"), results_b.add_prefix("combined_")], axis=1
    )
    out.to_csv(reports_dir / "multi_retailer_comparison.csv", index=False)

    print(f"\nDone in {time.time()-t0:.1f}s. Saved model + comparison CSV.")


if __name__ == "__main__":
    main()
