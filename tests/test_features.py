import pandas as pd
import pytest

from src.features import add_lag_features, add_rolling_features, assert_no_leakage, build_feature_matrix


def _make_synthetic_df(n_days=40):
    """Two stores, deterministic Sales = base + day_offset, so lag/rolling
    values can be hand-verified exactly."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store, base in [(1, 100), (2, 200)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "DayOfWeek": date.dayofweek + 1,
                    "Sales": base + i,
                    "Open": 1,
                    "Promo": int(i % 3 == 0),
                    "is_state_holiday": 0,
                    "SchoolHoliday": 0,
                }
            )
    return pd.DataFrame(rows)


def test_lag_7_hand_computed():
    df = _make_synthetic_df()
    df = add_lag_features(df)
    row = df[(df["Store"] == 1) & (df["Date"] == "2024-01-11")].iloc[0]  # i=10
    assert row["lag_7"] == pytest.approx(103.0)  # Sales[3] = 100 + 3


def test_lag_7_nan_before_warmup():
    df = _make_synthetic_df()
    df = add_lag_features(df)
    row = df[(df["Store"] == 1) & (df["Date"] == "2024-01-05")].iloc[0]  # i=4
    assert pd.isna(row["lag_7"])


def test_roll_mean_7_hand_computed():
    df = _make_synthetic_df()
    df = add_rolling_features(df)
    row = df[(df["Store"] == 1) & (df["Date"] == "2024-01-11")].iloc[0]  # i=10
    # mean(Sales[3:10]) = mean(103..109) = 106.0
    assert row["roll_mean_7"] == pytest.approx(106.0)


def test_roll_mean_7_nan_before_warmup():
    df = _make_synthetic_df()
    df = add_rolling_features(df)
    row = df[(df["Store"] == 1) & (df["Date"] == "2024-01-07")].iloc[0]  # i=6
    assert pd.isna(row["roll_mean_7"])


def test_build_feature_matrix_no_cross_store_bleed():
    """Store 2's lag_7 must never equal a value drawn from Store 1's series."""
    df = _make_synthetic_df()
    full = build_feature_matrix(df)
    store2_early = full[(full["Store"] == 2) & full["lag_7"].notna()].iloc[0]
    assert store2_early["lag_7"] >= 200  # Store 2's Sales all start at 200+


def test_assert_no_leakage_passes_on_correct_data():
    df = _make_synthetic_df()
    full = build_feature_matrix(df)
    assert assert_no_leakage(full, sample_size=10) is True


def test_assert_no_leakage_catches_corrupted_feature():
    df = _make_synthetic_df()
    full = build_feature_matrix(df)
    corrupt_idx = full.index[full["lag_7"].notna()][0]
    full.loc[corrupt_idx, "lag_7"] = 999999.0
    with pytest.raises(AssertionError):
        assert_no_leakage(full, sample_size=len(full))


def test_is_trainable_requires_open_and_complete_features():
    df = _make_synthetic_df()
    full = build_feature_matrix(df)
    early_row = full[(full["Store"] == 1) & (full["Date"] == "2024-01-05")].iloc[0]
    assert early_row["is_trainable"] == False  # noqa: E712 (incomplete lag_28)
