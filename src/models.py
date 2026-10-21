"""Baseline predictors and model-builder functions.

Baselines are zero-training: they just read an already-computed feature
column. Model builders return unfit sklearn-compatible estimators/pipelines.
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor


def seasonal_naive_predict(df):
    return df["lag_7"].to_numpy()


def moving_average_predict(df):
    return df["roll_mean_7"].to_numpy()


def set_categorical_dtypes(df, categorical_cols):
    """LightGBM's sklearn API auto-detects pandas `category` dtype columns
    at fit time, so cast the known categoricals before training/predicting."""
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df


def build_ridge_pipeline(numeric_cols, categorical_cols, alpha=1.0, random_state=42):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )


def build_lightgbm_model(**overrides):
    params = dict(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    params.update(overrides)
    return LGBMRegressor(**params)
