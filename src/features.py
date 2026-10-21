"""Calendar, lag, and rolling-window feature engineering with leakage safeguards.

Design rule: lag/rolling features are always computed by shifting the target
*before* windowing, so a feature for row (store, date) only ever touches rows
with an earlier date for that same store. `assert_no_leakage` independently
re-derives a sample of these values from raw per-store history to prove it.
"""

import numpy as np
import pandas as pd

LAG_COLS = ["lag_1", "lag_7", "lag_14", "lag_28"]
ROLL_COLS = [
    "roll_mean_7", "roll_std_7",
    "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
    "roll_min_7", "roll_max_7",
    "promo_last_7_days_count",
]
REQUIRED_FEATURE_COLS = LAG_COLS + ROLL_COLS
CALENDAR_COLS = [
    "Year", "Month", "Day", "WeekOfYear", "Quarter", "is_weekend",
    "is_month_start", "is_month_end", "is_day_before_holiday", "is_day_after_holiday",
]

# Shared modeling contract, reused identically by notebooks 04-08 so every
# model/baseline/evaluation/forecast step trains and predicts on the same
# column set.
CATEGORICAL_FEATURE_COLS = ["Store", "StoreType", "Assortment", "state_holiday_type", "DayOfWeek"]
NUMERIC_FEATURE_COLS = (
    [
        "CompetitionDistance", "competition_open_days", "competition_distance_missing",
        "has_competition_since_info", "has_competitor_open", "Promo", "SchoolHoliday",
        "is_state_holiday", "promo2_active",
    ]
    + LAG_COLS + ROLL_COLS + CALENDAR_COLS
)
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


def add_calendar_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    # Holiday calendars are known in advance (exogenous), so looking 1 day
    # ahead/behind here is not target leakage -- unlike the Sales-derived
    # lag/rolling features below, which must only look backward.
    holiday_flag = df["is_state_holiday"].astype(int)
    df["is_day_before_holiday"] = (
        holiday_flag.groupby(df["Store"]).shift(-1).fillna(0).astype(int)
    )
    df["is_day_after_holiday"] = (
        holiday_flag.groupby(df["Store"]).shift(1).fillna(0).astype(int)
    )
    return df


def add_lag_features(df, target_col="Sales", group_col="Store", lags=(1, 7, 14, 28)):
    df = df.copy()
    grouped_target = df.groupby(group_col)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grouped_target.shift(lag)
    return df


def add_rolling_features(df, target_col="Sales", group_col="Store", windows=(7, 14, 28)):
    df = df.copy()
    shifted = df.groupby(group_col)[target_col].shift(1)

    for window in windows:
        roll = shifted.groupby(df[group_col]).rolling(window=window, min_periods=window)
        df[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)

    roll7 = shifted.groupby(df[group_col]).rolling(window=7, min_periods=7)
    df["roll_min_7"] = roll7.min().reset_index(level=0, drop=True)
    df["roll_max_7"] = roll7.max().reset_index(level=0, drop=True)

    shifted_promo = df.groupby(group_col)["Promo"].shift(1)
    promo_roll7 = shifted_promo.groupby(df[group_col]).rolling(window=7, min_periods=7)
    df["promo_last_7_days_count"] = promo_roll7.sum().reset_index(level=0, drop=True)
    return df


def build_feature_matrix(df, target_col="Sales", group_col="Store"):
    """Returns the FULL engineered frame, including closed-store and
    incomplete-history rows (needed by `assert_no_leakage` and by the
    recursive test-horizon forecast in notebook 08). Call
    `filter_trainable_rows` to get the model-ready subset.
    """
    df = df.sort_values([group_col, "Date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col=target_col, group_col=group_col)
    df = add_rolling_features(df, target_col=target_col, group_col=group_col)

    if target_col in df.columns:
        df["log_sales"] = np.log1p(df[target_col].clip(lower=0))

    df["is_trainable"] = (df["Open"] == 1) & df[REQUIRED_FEATURE_COLS].notna().all(axis=1)
    return df


def filter_trainable_rows(df):
    return df[df["is_trainable"]].reset_index(drop=True)


def assert_no_leakage(df, date_col="Date", group_col="Store", target_col="Sales",
                       sample_size=20, seed=42):
    """Independently recompute lag_7/roll_mean_7 for a random sample of rows,
    using only rows with an earlier date for the same group, and assert they
    match the stored feature columns. `df` must be the FULL engineered frame
    (pre-filter), so per-store history is complete.
    """
    rng = np.random.default_rng(seed)
    candidates = df.index[df["lag_7"].notna() & df["roll_mean_7"].notna()].to_numpy()
    if len(candidates) == 0:
        raise AssertionError("No rows with complete lag_7/roll_mean_7 to check")
    sample_idx = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)

    checked = 0
    for idx in sample_idx:
        row = df.loc[idx]
        group_value, date = row[group_col], row[date_col]
        history = df[(df[group_col] == group_value) & (df[date_col] < date)].sort_values(date_col)
        if len(history) < 7:
            continue
        expected_lag_7 = history[target_col].iloc[-7]
        expected_roll_mean_7 = history[target_col].iloc[-7:].mean()
        if not np.isclose(row["lag_7"], expected_lag_7, equal_nan=True):
            raise AssertionError(
                f"Leakage check failed at {group_col}={group_value}, {date_col}={date}: "
                f"lag_7={row['lag_7']} != recomputed {expected_lag_7}"
            )
        if not np.isclose(row["roll_mean_7"], expected_roll_mean_7, equal_nan=True):
            raise AssertionError(
                f"Leakage check failed at {group_col}={group_value}, {date_col}={date}: "
                f"roll_mean_7={row['roll_mean_7']} != recomputed {expected_roll_mean_7}"
            )
        checked += 1

    if checked == 0:
        raise AssertionError("No sampled row had enough history to verify")
    return True
