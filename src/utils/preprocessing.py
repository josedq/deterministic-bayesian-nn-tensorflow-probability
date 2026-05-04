"""sklearn ColumnTransformer used for all training and evaluation."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.constants import (
    CATEGORICAL_FEATURES,
    NUMERICAL_REVIEW_FEATURES,
    NUMERICAL_STANDARD_FEATURES,
)

def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer (median/zero imputation + scaling + OHE)."""
    standard_numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    review_numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler',  StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ('num',     standard_numeric, NUMERICAL_STANDARD_FEATURES),
            ('reviews', review_numeric,   NUMERICAL_REVIEW_FEATURES),
            ('cat',     categorical,      CATEGORICAL_FEATURES),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
