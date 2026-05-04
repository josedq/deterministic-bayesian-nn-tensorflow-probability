"""Feature engineering and data loading shared across experiment scripts."""

import numpy as np
import pandas as pd
from datetime import datetime

from config.constants import TARGET, TARGET_CAP, COLS_TO_DROP

def count_items(text):
    """Number of comma-separated items in a string cell (e.g. amenities)."""
    if pd.isna(text):
        return 0
    return len(str(text).split(','))

def days_since(date_str):
    """Days elapsed since date_str; returns NaN if unparseable."""
    if pd.isna(date_str):
        return np.nan
    try:
        return (datetime.now() - pd.to_datetime(date_str)).days
    except Exception:
        return np.nan

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Parse price, filter by cap, drop irrelevant columns, compute derived features."""
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(
        df[TARGET].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce',
    )
    df = df.dropna(subset=[TARGET])
    df = df[(df[TARGET] > 0) & (df[TARGET] <= TARGET_CAP)]

    df = df.drop(columns=COLS_TO_DROP, errors='ignore')

    df['amenities_count']         = df['amenities'].apply(count_items)
    df['host_days_active']        = df['host_since'].apply(days_since)
    df['days_since_first_review'] = df['first_review'].apply(days_since)
    df['days_since_last_review']  = df['last_review'].apply(days_since)

    for col in ['host_is_superhost', 'host_identity_verified', 'instant_bookable']:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

    return df

def load_listings(path: str) -> pd.DataFrame:
    """Load a raw listings CSV and apply the feature engineering pipeline."""
    return apply_feature_engineering(pd.read_csv(path))
