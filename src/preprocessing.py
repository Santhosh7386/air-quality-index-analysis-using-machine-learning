"""
preprocessing.py
-----------------
Shared preprocessing logic used both at initial training time and whenever
the admin panel retrains a country's model with newly uploaded data.

Keeping this logic in one place guarantees that a model trained today and
a model retrained next month after new data is merged in are built using
exactly the same feature engineering steps.
"""

import pandas as pd

# The raw columns every dataset (original or admin-uploaded) must contain
REQUIRED_COLUMNS = [
    "Country", "City", "Season",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "AQI_Value", "AQI_Category",
]

NUMERIC_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
TARGET = "AQI_Value"


def validate_uploaded_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that an admin-uploaded CSV has the required columns and
    sensible values. Returns (is_valid, message).
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"

    if df.empty:
        return False, "The uploaded file contains no rows."

    # Numeric columns must actually be numeric (or convertible / blank)
    for col in NUMERIC_FEATURES + [TARGET]:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Allow missing values in pollutant columns (they get imputed later),
        # but the target AQI_Value must always be present and numeric.
        if col == TARGET and converted.isnull().any():
            return False, f"Column '{col}' must be numeric with no missing values."

    if not df["Season"].isin(SEASONS).all():
        bad = sorted(set(df["Season"].unique()) - set(SEASONS))
        return False, f"Season column contains invalid values: {bad}. Must be one of {SEASONS}."

    return True, "Data validated successfully."


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning + feature engineering pipeline to a raw
    country-level dataframe and return a model-ready dataframe.

    Steps:
      1. Coerce pollutant columns to numeric.
      2. Impute missing pollutant readings with the column median
         (median is robust to the outliers present in pollutant data).
      3. One-hot encode the 'Season' categorical feature.
    """
    data = df.copy()

    for col in NUMERIC_FEATURES + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Median imputation for missing pollutant readings
    for col in NUMERIC_FEATURES:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)

    # Drop any row where the target itself is missing (should not happen
    # after validation, but kept as a safety net)
    data = data.dropna(subset=[TARGET])

    # One-hot encode Season, ensuring all four season columns always exist
    # even if a particular country/season split doesn't include every season.
    season_dummies = pd.get_dummies(data["Season"], prefix="Season")
    for season in SEASONS:
        col_name = f"Season_{season}"
        if col_name not in season_dummies.columns:
            season_dummies[col_name] = 0
    season_dummies = season_dummies[[f"Season_{s}" for s in SEASONS]]

    data = pd.concat([data.reset_index(drop=True), season_dummies.reset_index(drop=True)], axis=1)

    return data


def get_feature_columns() -> list:
    """The final list of feature columns fed into the ML model, in fixed order."""
    return NUMERIC_FEATURES + [f"Season_{s}" for s in SEASONS]
