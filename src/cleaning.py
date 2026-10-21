"""Merge, type-fix, and derive leakage-safe exogenous features from raw Rossmann data.

`Open == 0` rows are deliberately kept (not dropped) so each store's date index
stays continuous -- dropping them here would corrupt the lag/rolling-window
arithmetic computed later in `features.py`.
"""

import pandas as pd

MONTH_ABBR_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def merge_train_store(df, store):
    before = len(df)
    merged = df.merge(store, on="Store", how="left")
    assert len(merged) == before, "merge changed row count"
    return merged


def fill_missing_open(df):
    """test.csv has ~11 rows with a missing Open flag; historically treated as open."""
    df = df.copy()
    df["Open"] = df["Open"].fillna(1).astype(int)
    return df


def encode_state_holiday(df):
    df = df.copy()
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0.0": "0", "nan": "0"})
    df["state_holiday_type"] = pd.Categorical(df["StateHoliday"])
    df["is_state_holiday"] = (df["StateHoliday"] != "0").astype(int)
    return df


def impute_competition_fields(df):
    df = df.copy()
    median_dist = df["CompetitionDistance"].median()
    df["competition_distance_missing"] = df["CompetitionDistance"].isna().astype(int)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_dist)

    df["has_competition_since_info"] = (
        df["CompetitionOpenSinceMonth"].notna() & df["CompetitionOpenSinceYear"].notna()
    ).astype(int)
    return df


def derive_competition_features(df):
    df = df.copy()
    has_info = df["has_competition_since_info"] == 1

    df["CompetitionOpenDate"] = pd.NaT
    if has_info.any():
        comp_open_date = pd.to_datetime(
            {
                "year": df.loc[has_info, "CompetitionOpenSinceYear"].astype(int),
                "month": df.loc[has_info, "CompetitionOpenSinceMonth"].astype(int),
                "day": 1,
            },
            errors="coerce",
        )
        df.loc[has_info, "CompetitionOpenDate"] = comp_open_date.values

    days_open = (df["Date"] - df["CompetitionOpenDate"]).dt.days
    df["competition_open_days"] = days_open.clip(lower=0).fillna(0).astype(int)
    df["has_competitor_open"] = (
        has_info & (df["Date"] >= df["CompetitionOpenDate"])
    ).astype(int)
    return df


def _promo2_active_row(row):
    if row["Promo2"] != 1 or pd.isna(row["PromoInterval"]):
        return 0
    since_year, since_week = row.get("Promo2SinceYear"), row.get("Promo2SinceWeek")
    if pd.isna(since_year) or pd.isna(since_week):
        return 0
    iso_year, iso_week, _ = row["Date"].isocalendar()
    started = (iso_year > since_year) or (iso_year == since_year and iso_week >= since_week)
    if not started:
        return 0
    active_months = {MONTH_ABBR_TO_NUM[m] for m in row["PromoInterval"].split(",")}
    return int(row["Date"].month in active_months)


def derive_promo2_active(df):
    df = df.copy()
    df["promo2_active"] = df.apply(_promo2_active_row, axis=1).astype(int)
    return df


def clean_pipeline(df, store):
    """Full cleaning pipeline, reused identically on train and test data.

    Row count is preserved except for the merge-column additions -- no rows
    are dropped here; `features.build_feature_matrix` handles row filtering.
    """
    df = merge_train_store(df, store)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    if df["Open"].isna().any():
        df = fill_missing_open(df)
    else:
        df["Open"] = df["Open"].astype(int)
    df = encode_state_holiday(df)
    df = impute_competition_fields(df)
    df = derive_competition_features(df)
    df = derive_promo2_active(df)
    return df
