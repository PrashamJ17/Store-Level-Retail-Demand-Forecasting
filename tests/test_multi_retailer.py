import numpy as np
import pandas as pd
import pytest

from src.multi_retailer import (
    add_weekly_features,
    aggregate_to_weekly,
    assert_no_leakage_weekly,
)


def _make_synthetic_daily(n_days=70):
    """Two sources/stores, deterministic sales = base + day_offset, starting
    Monday 2024-01-01 so each 7-day block is a clean calendar week -- lets
    weekly sums and lag/rolling values be hand-verified exactly."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store_id, base in [("source_a_1", 100), ("source_b_1", 500)]:
        for i, date in enumerate(dates):
            rows.append({
                "combined_store_id": store_id,
                "dataset_source": store_id.rsplit("_", 1)[0],
                "Date": date,
                "sales": base + i,
                "promo_flag": int(i % 5 == 0),
                "holiday_flag": 0,
                "store_type_combined": f"{store_id}_type",
                "competition_distance": np.nan,
                "store_size": np.nan,
                "temperature": np.nan,
                "fuel_price": np.nan,
                "cpi": np.nan,
                "unemployment": np.nan,
            })
    return pd.DataFrame(rows)


def test_aggregate_to_weekly_sums_correctly():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    store_a = weekly[weekly["combined_store_id"] == "source_a_1"].reset_index(drop=True)
    # week 0: days 0-6, sales 100..106 -> sum = 721
    assert store_a.loc[0, "weekly_sales"] == pytest.approx(721)
    # week 1: days 7-13, sales 107..113 -> sum = 770
    assert store_a.loc[1, "weekly_sales"] == pytest.approx(770)


def test_no_cross_store_bleed_in_weekly_aggregation():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    store_b_week0 = weekly[weekly["combined_store_id"] == "source_b_1"].iloc[0]
    assert store_b_week0["weekly_sales"] >= 500 * 7  # base=500, must not mix with source_a's base=100


def test_lag_1_and_roll_mean_4_hand_computed():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    full = add_weekly_features(weekly)
    store_a = full[full["combined_store_id"] == "source_a_1"].reset_index(drop=True)

    # week index 5 (0-based): lag_1 = week4's total = 721 + 49*4 = 917
    assert store_a.loc[5, "lag_1"] == pytest.approx(917)
    # roll_mean_4 = mean(week1..week4) = mean(770, 819, 868, 917) = 843.5
    assert store_a.loc[5, "roll_mean_4"] == pytest.approx(843.5)


def test_lag_1_nan_before_warmup():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    full = add_weekly_features(weekly)
    store_a = full[full["combined_store_id"] == "source_a_1"].reset_index(drop=True)
    assert pd.isna(store_a.loc[0, "lag_1"])


def test_assert_no_leakage_weekly_passes_on_correct_data():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    full = add_weekly_features(weekly)
    assert assert_no_leakage_weekly(full, sample_size=10) is True


def test_assert_no_leakage_weekly_catches_corrupted_feature():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    full = add_weekly_features(weekly)
    corrupt_idx = full.index[full["lag_1"].notna() & full["roll_mean_4"].notna()][0]
    full.loc[corrupt_idx, "lag_1"] = 999999.0
    with pytest.raises(AssertionError):
        assert_no_leakage_weekly(full, sample_size=len(full))


def test_has_promo_is_any_day_in_week():
    df = _make_synthetic_daily()
    weekly = aggregate_to_weekly(df)
    store_a = weekly[weekly["combined_store_id"] == "source_a_1"].reset_index(drop=True)
    # promo_flag is 1 on day i % 5 == 0 -> every week of 7 days has at least one such day
    assert (store_a["has_promo"] == 1).all()
