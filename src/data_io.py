"""Thin I/O helpers for reading raw Kaggle CSVs and caching intermediate frames as parquet."""

from pathlib import Path

import pandas as pd

RAW_DTYPES_TRAIN = {
    "Store": "int32",
    "DayOfWeek": "int8",
    "Sales": "int32",
    "Customers": "int32",
    "Open": "int8",
    "Promo": "int8",
    "StateHoliday": "string",
    "SchoolHoliday": "int8",
}

# test.csv has a handful of missing `Open` values upstream, so it needs a
# nullable float dtype rather than train.csv's int8.
RAW_DTYPES_TEST = {
    "Id": "int32",
    "Store": "int32",
    "DayOfWeek": "int8",
    "Open": "float64",
    "Promo": "int8",
    "StateHoliday": "string",
    "SchoolHoliday": "int8",
}


def read_raw_csvs(raw_dir):
    """Load train.csv, test.csv, and store.csv from `raw_dir` with explicit dtypes."""
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv", dtype=RAW_DTYPES_TRAIN, parse_dates=["Date"])
    test = pd.read_csv(raw_dir / "test.csv", dtype=RAW_DTYPES_TEST, parse_dates=["Date"])
    store = pd.read_csv(raw_dir / "store.csv")
    return train, test, store


def save_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    return pd.read_parquet(path)
