"""Shared plotting helpers for consistent EDA/interpretation visuals."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

sns.set_theme(style="whitegrid")


def decomposition_plot(series, period=7, title="Seasonal-Trend Decomposition"):
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title=f"{title} — Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    fig.tight_layout()
    return fig, result


def promo_effect_plot(df, sales_col="Sales", promo_col="Promo"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x=promo_col, y=sales_col, ax=ax, errorbar="sd")
    ax.set_xticklabels(["No Promo", "Promo"])
    ax.set_title("Mean Sales: Promo vs No Promo")
    return fig


def save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
