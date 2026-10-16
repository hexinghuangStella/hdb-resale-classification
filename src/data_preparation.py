"""Deterministic cleaning and feature engineering shared by EDA and training."""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = ["floor_area_sqm", "lease_commence_date", "remaining_lease_months", "sale_year", "sale_month", "storey_midpoint"]
CATEGORICAL_FEATURES = ["flat_type", "block", "street_name", "town_id", "flatm_id"]
TARGET_LABELS = {"Above Median": 1, "Below Median": 0}


def parse_remaining_lease(value):
    """Convert numeric years or '70 years 03 months' to months."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.fullmatch(r"(?:(\d+)\s*years?)?\s*(?:(\d+)\s*months?)?", text)
    if match and any(match.groups()):
        return int(match.group(1) or 0) * 12 + int(match.group(2) or 0)
    try:
        years = float(text)
        return years * 12 if np.isfinite(years) and years >= 0 else np.nan
    except ValueError:
        return np.nan


def clean_data(df):
    """Normalize labels, remove exact duplicates, and validate the target."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)
    if df.columns.duplicated().any():
        raise ValueError("Column names are duplicated after normalization.")
    df = df.dropna(how="all").drop_duplicates()
    if "price_category" not in df:
        raise ValueError("Missing target column: price_category")
    labels = df["price_category"].astype("string").str.strip()
    invalid = labels.isna() | ~labels.isin(TARGET_LABELS)
    if invalid.any():
        raise ValueError(f"Missing or unknown price_category values: {labels[invalid].unique().tolist()}")
    df["price_category"] = labels.map(TARGET_LABELS).astype(int)
    return df.reset_index(drop=True)


def load_data(path):
    return clean_data(pd.read_csv(path))


def engineer_features(df):
    """Apply row-local transforms; no statistics are learned from held-out data.

    Stable town/model IDs are categorical features. Redundant names and record
    IDs are excluded, avoiding name lookups learned across the full dataset.
    """
    required = ["floor_area_sqm", "lease_commence_date", "remaining_lease", "month", "storey_range", *CATEGORICAL_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    result = pd.DataFrame(index=df.index)
    for col in ["floor_area_sqm", "lease_commence_date"]:
        result[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["remaining_lease_months"] = df["remaining_lease"].map(parse_remaining_lease)
    dates = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    result["sale_year"] = dates.dt.year
    result["sale_month"] = dates.dt.month
    storeys = df["storey_range"].astype("string").str.extract(r"^\s*(\d+)\s+TO\s+(\d+)\s*$")
    result["storey_midpoint"] = storeys.astype(float).mean(axis=1)
    for col in CATEGORICAL_FEATURES:
        values = df[col].astype("string").str.strip().str.upper().replace("", pd.NA)
        if col == "flat_type":
            values = values.replace({"FOUR ROOM": "4 ROOM"})
        result[col] = values.astype(object).where(values.notna(), np.nan)
    return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def split_data(df, config):
    validation_size = float(config["validation_size"])
    test_size = float(config["test_size"])
    if not (0 < validation_size < 1 and 0 < test_size < 1 and validation_size + test_size < 1):
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")
    X = engineer_features(df)
    y = df["price_category"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=validation_size + test_size,
        random_state=config["random_state"], stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_size / (validation_size + test_size),
        random_state=config["random_state"], stratify=y_holdout,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
