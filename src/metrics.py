"""Forecast accuracy metrics: RMSE, MAE, WMAPE, and MASE.

MASE and WMAPE aren't in scikit-learn, so both are implemented here. MASE in
particular must be computed carefully for a *global* multi-store model: naive
(lag-7) errors used to scale it are computed per store (`train_groups`) and
then pooled, so the scale is never contaminated by diffing across a store
boundary in the concatenated training array.
"""

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wmape(y_true, y_pred):
    """Weighted MAPE: sum(|error|) / sum(|actual|). Avoids the blow-up plain
    MAPE suffers on near-zero-sales days (e.g. closed stores)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mase(y_true, y_pred, y_train, seasonal_period=7, train_groups=None):
    """Mean Absolute Scaled Error, scaled by the seasonal-naive MAE on
    *training* data only (never on validation data, to avoid leaking
    validation information into the metric itself).

    `y_train` (and `train_groups`, if given) must already be sorted by
    (group, date) -- the scale is a positional diff, like the lag features.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if train_groups is None:
        if len(y_train) <= seasonal_period:
            raise ValueError("y_train must have more than seasonal_period observations")
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    else:
        train_groups = np.asarray(train_groups)
        chunks = []
        for group in np.unique(train_groups):
            series = y_train[train_groups == group]
            if len(series) > seasonal_period:
                chunks.append(np.abs(series[seasonal_period:] - series[:-seasonal_period]))
        if not chunks:
            raise ValueError("No group had more than seasonal_period observations")
        naive_errors = np.concatenate(chunks)

    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def evaluate_all(y_true, y_pred, y_train, seasonal_period=7, train_groups=None):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, seasonal_period=seasonal_period, train_groups=train_groups),
    }
