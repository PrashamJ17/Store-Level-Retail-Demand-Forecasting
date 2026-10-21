"""Strictly date-ordered walk-forward (expanding-window) cross-validation splits."""

import pandas as pd


def walk_forward_splits(dates, n_folds=5, val_days=42):
    """Return a list of dicts with boolean train/val masks, one per fold.

    Validation blocks are `val_days` long, non-overlapping, and built
    backward from the data's maximum date; each fold's training window
    expands to include every earlier date. No shuffling -- every mask is a
    pure function of `dates`.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    max_date = dates.max()

    val_blocks = []
    val_end = max_date
    for _ in range(n_folds):
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        val_blocks.append((val_start, val_end))
        val_end = val_start - pd.Timedelta(days=1)
    val_blocks.reverse()

    splits = []
    for fold_idx, (val_start, val_end) in enumerate(val_blocks, start=1):
        train_end = val_start - pd.Timedelta(days=1)
        train_mask = (dates <= train_end).to_numpy()
        val_mask = ((dates >= val_start) & (dates <= val_end)).to_numpy()
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        splits.append(
            {
                "fold": fold_idx,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "train_mask": train_mask,
                "val_mask": val_mask,
            }
        )
    return splits
