import numpy as np
import pytest

from src.metrics import evaluate_all, mae, mase, rmse, wmape


def test_rmse_hand_computed():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # errors: 2, -2, 3 -> squared: 4, 4, 9 -> mean 17/3 -> sqrt
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(17 / 3))


def test_mae_hand_computed():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)


def test_wmape_hand_computed():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    assert wmape(y_true, y_pred) == pytest.approx(7 / 60)


def test_wmape_zero_actuals_is_nan():
    assert np.isnan(wmape([0, 0], [1, 2]))


def test_mase_seasonal_naive_self_check():
    """The seasonal-naive forecast's own MASE should be ~1.0 by construction
    -- this validates the mase() implementation itself."""
    rng = np.random.default_rng(0)
    y_train = 100 + 5 * np.sin(np.arange(70) * 2 * np.pi / 7) + rng.normal(0, 0.01, 70)
    y_true = y_train[7:]
    y_pred_seasonal_naive = y_train[:-7]
    result = mase(y_true, y_pred_seasonal_naive, y_train, seasonal_period=7)
    assert result == pytest.approx(1.0, abs=0.05)


def test_mase_per_group_scale_not_contaminated_by_group_boundary():
    """Scale must be computed within each store, not across the concatenation
    boundary between stores -- otherwise a global model's MASE is wrong."""
    store_a = np.array([100.0] * 40)  # flat series -> naive errors all 0
    store_b = np.array([0.0, 100.0] * 20)  # alternating -> huge naive errors
    y_train = np.concatenate([store_a, store_b])
    groups = np.array(["A"] * 40 + ["B"] * 40)

    y_true = np.array([100.0, 100.0])
    y_pred = np.array([100.0, 100.0])

    # Scale from store A alone (flat) would be 0 -> undefined/huge MASE.
    with pytest.raises(ValueError):
        mase(y_true, y_pred, store_a[store_a == 0.0], seasonal_period=7)

    pooled = mase(y_true, y_pred, y_train, seasonal_period=7, train_groups=groups)
    assert pooled == pytest.approx(0.0)  # prediction is perfect either way


def test_evaluate_all_returns_expected_keys():
    y_train = np.arange(1, 71, dtype=float)
    result = evaluate_all([10.0], [12.0], y_train, seasonal_period=7)
    assert set(result.keys()) == {"rmse", "mae", "wmape", "mase"}
