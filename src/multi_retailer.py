"""Unify Rossmann (daily) and Walmart (weekly) store-sales data onto a
common store x week grain, for a single cross-retailer model.

Design: each source gets a thin `prep_*` function that maps its own raw
columns onto one shared intermediate schema (combined_store_id,
dataset_source, Date, sales, promo_flag, holiday_flag, store_type_combined,
plus whatever source-only numeric columns it has -- left NaN for the other
source, which LightGBM treats as native missing). `aggregate_to_weekly` and
`add_weekly_features` then run identically on either source's prepared
frame, or on the two concatenated together.

Rossmann and Walmart both number stores 1..N, so `combined_store_id` is
dataset-prefixed (e.g. "rossmann_262", "walmart_12") to avoid collisions --
every lag/rolling feature and the leakage check below group by this column,
never by the raw Store number.
"""

import numpy as np
import pandas as pd

COMMON_COLS = [
    "combined_store_id", "dataset_source", "Date", "sales", "promo_flag",
    "holiday_flag", "store_type_combined", "competition_distance",
    "store_size", "temperature", "fuel_price", "cpi", "unemployment",
]

LAG_WEEKS = (1, 4, 8)
ROLL_WEEKS = (4, 8)


def prep_rossmann(train_clean):
    """Map Rossmann's cleaned daily frame onto the common intermediate schema."""
    df = train_clean.copy()
    out = pd.DataFrame({
        "combined_store_id": "rossmann_" + df["Store"].astype(str),
        "dataset_source": "rossmann",
        "Date": df["Date"],
        "sales": df["Sales"],
        "promo_flag": df["Promo"],
        "holiday_flag": df["is_state_holiday"],
        "store_type_combined": "rossmann_" + df["StoreType"].astype(str),
        "competition_distance": df["CompetitionDistance"],
        "store_size": np.nan,
        "temperature": np.nan,
        "fuel_price": np.nan,
        "cpi": np.nan,
        "unemployment": np.nan,
    })
    return out[COMMON_COLS]


def prep_walmart(walmart_raw):
    """Map the Walmart Sales CSV (Store, Date, Weekly_Sales, Holiday_Flag,
    Temperature, Fuel_Price, CPI, Unemployment -- already weekly grain, no
    Dept/MarkDown/StoreType/Size columns in this file) onto the common
    intermediate schema. No promo signal exists in this dataset, so
    promo_flag is left NaN (unknown) rather than fabricated as 0/false;
    store_type_combined is a single constant bucket for the same reason --
    there is no real per-store type field to carry."""
    df = walmart_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    out = pd.DataFrame({
        "combined_store_id": "walmart_" + df["Store"].astype(str),
        "dataset_source": "walmart",
        "Date": df["Date"],
        "sales": df["Weekly_Sales"],
        "promo_flag": np.nan,
        "holiday_flag": df["Holiday_Flag"],
        "store_type_combined": "walmart_store",
        "competition_distance": np.nan,
        "store_size": np.nan,
        "temperature": df["Temperature"],
        "fuel_price": df["Fuel_Price"],
        "cpi": df["CPI"],
        "unemployment": df["Unemployment"],
    })
    return out[COMMON_COLS]


def aggregate_to_weekly(df):
    """One row per (combined_store_id, week): sales summed, flags OR'd
    (`max`), store attributes carried via `first` (constant per store),
    external numeric signals averaged over the week."""
    df = df.copy()
    df["week_start"] = df["Date"].dt.to_period("W").dt.start_time

    agg = df.groupby(["combined_store_id", "week_start"], as_index=False).agg(
        dataset_source=("dataset_source", "first"),
        weekly_sales=("sales", "sum"),
        has_promo=("promo_flag", "max"),
        has_holiday=("holiday_flag", "max"),
        store_type_combined=("store_type_combined", "first"),
        competition_distance=("competition_distance", "first"),
        store_size=("store_size", "first"),
        temperature=("temperature", "mean"),
        fuel_price=("fuel_price", "mean"),
        cpi=("cpi", "mean"),
        unemployment=("unemployment", "mean"),
    )
    return agg.sort_values(["combined_store_id", "week_start"]).reset_index(drop=True)


WEEKLY_LAG_COLS = [f"lag_{w}" for w in LAG_WEEKS]
WEEKLY_ROLL_COLS = [c for w in ROLL_WEEKS for c in (f"roll_mean_{w}", f"roll_std_{w}")]
WEEKLY_REQUIRED_FEATURE_COLS = WEEKLY_LAG_COLS + WEEKLY_ROLL_COLS


def add_weekly_features(weekly_df):
    """Lag/rolling features computed per combined_store_id, shift-before-roll
    (same leakage-prevention rule as the daily Rossmann pipeline in
    src/features.py), plus simple calendar features from week_start."""
    df = weekly_df.sort_values(["combined_store_id", "week_start"]).reset_index(drop=True)

    grouped_target = df.groupby("combined_store_id")["weekly_sales"]
    for w in LAG_WEEKS:
        df[f"lag_{w}"] = grouped_target.shift(w)

    shifted = df.groupby("combined_store_id")["weekly_sales"].shift(1)
    for w in ROLL_WEEKS:
        roll = shifted.groupby(df["combined_store_id"]).rolling(window=w, min_periods=w)
        df[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = roll.std().reset_index(level=0, drop=True)

    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)

    if "weekly_sales" in df.columns:
        df["log_weekly_sales"] = np.log1p(df["weekly_sales"].clip(lower=0))

    df["is_trainable"] = df[WEEKLY_REQUIRED_FEATURE_COLS].notna().all(axis=1)
    return df


def filter_trainable_weekly(df):
    return df[df["is_trainable"]].reset_index(drop=True)


def assert_no_leakage_weekly(df, sample_size=20, seed=42):
    """Same independent-recompute proof as src.features.assert_no_leakage,
    adapted to the weekly grain and combined_store_id grouping. `df` must be
    the FULL (pre-filter) weekly frame so per-store history is complete."""
    rng = np.random.default_rng(seed)
    candidates = df.index[df["lag_1"].notna() & df["roll_mean_4"].notna()].to_numpy()
    if len(candidates) == 0:
        raise AssertionError("No rows with complete lag_1/roll_mean_4 to check")
    sample_idx = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)

    checked = 0
    for idx in sample_idx:
        row = df.loc[idx]
        store_id, week = row["combined_store_id"], row["week_start"]
        history = df[(df["combined_store_id"] == store_id) & (df["week_start"] < week)].sort_values("week_start")
        if len(history) < 4:
            continue
        expected_lag_1 = history["weekly_sales"].iloc[-1]
        expected_roll_mean_4 = history["weekly_sales"].iloc[-4:].mean()
        if not np.isclose(row["lag_1"], expected_lag_1, equal_nan=True):
            raise AssertionError(f"Leakage check failed at {store_id}, {week}: lag_1 mismatch")
        if not np.isclose(row["roll_mean_4"], expected_roll_mean_4, equal_nan=True):
            raise AssertionError(f"Leakage check failed at {store_id}, {week}: roll_mean_4 mismatch")
        checked += 1

    if checked == 0:
        raise AssertionError("No sampled row had enough history to verify")
    return True
