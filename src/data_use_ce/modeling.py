"""Preprocessing and train/test split of the student table."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_use_ce.data import TARGET


def make_preprocessor(df: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    """Builds the encoder that turns the chosen columns into numbers.

    Numeric columns pass through unchanged: trees need no scaling, and a tree plot then shows
    the real units. Text columns are one-hot encoded, and a two-value column becomes a single
    0/1 column. The encoder is not fitted: put it in a pipeline with the model, so that it
    learns its categories from the training students only.

    Args:
        df: Students; only used to tell numeric columns from text columns.
        features: Columns to encode.

    Returns:
        An unfitted transformer that outputs a DataFrame with readable column names.
    """
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    text = [c for c in features if c not in numeric]
    encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    transformer = ColumnTransformer(
        [("text", encoder, text), ("numeric", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")


def split_students(
    df: pd.DataFrame, features: Sequence[str], test_size: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits students into a training and a test set with the same excellent share.

    Args:
        df: Students, with the "Excellent" column.
        features: Columns used as model input.
        test_size: Share of students kept aside to test the model.
        seed: Seed of the random split.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        df[list(features)],
        df[TARGET],
        test_size=test_size,
        stratify=df[TARGET],
        random_state=seed,
    )
