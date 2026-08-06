"""
train.py
--------
Trains one Gradient Boosting Regressor per country and saves it to disk.

Design rationale (kept simple on purpose for viva explanation):
  - Air pollution dynamics differ meaningfully between countries (different
    pollution sources, regulations, climates, and typical AQI ranges), so a
    single global model would blur these country-specific relationships.
    Training one model per country lets each model specialise.
  - Gradient Boosting Regressor is used because AQI_Value is a continuous
    number (regression, not classification) that is deterministically
    converted into a category afterwards using fixed EPA-style breakpoints
    (see src/aqi_utils.py). Gradient Boosting handles non-linear
    relationships between pollutants and AQI well, is robust to the mild
    outliers present in pollutant readings, and does not require feature
    scaling.
"""

import os
import json
from datetime import datetime

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from src.preprocessing import clean_and_engineer, get_feature_columns, TARGET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "countries")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs", "training_log.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _country_file(country: str) -> str:
    safe = country.replace(" ", "_")
    return os.path.join(DATA_DIR, f"{safe}.csv")


def _model_file(country: str) -> str:
    safe = country.replace(" ", "_")
    return os.path.join(MODEL_DIR, f"{safe}_model.pkl")


def load_country_data(country: str) -> pd.DataFrame:
    path = _country_file(country)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset found for country '{country}' at {path}")
    return pd.read_csv(path)


def train_country_model(country: str, df: pd.DataFrame = None) -> dict:
    """
    Train (or retrain) a Gradient Boosting model for a single country.

    Parameters
    ----------
    country : str
        Country name, used to locate/save the correct files.
    df : pd.DataFrame, optional
        If provided, this dataframe is used for training AND is saved back
        as the country's dataset (used by the admin retraining workflow
        after merging newly uploaded records). If not provided, the
        existing on-disk country CSV is loaded as-is.

    Returns
    -------
    dict with training metrics and metadata, which is also appended to the
    training log CSV.
    """
    if df is None:
        df = load_country_data(country)
    else:
        # Persist the (possibly merged) dataset back to disk so future
        # retraining and the dashboard reflect the newest data.
        df.to_csv(_country_file(country), index=False)

    processed = clean_and_engineer(df)
    feature_cols = get_feature_columns()

    X = processed[feature_cols]
    y = processed[TARGET]

    test_size = 0.2 if len(processed) >= 20 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Save model bundle: model + the exact feature column order it expects
    joblib.dump({"model": model, "feature_columns": feature_cols}, _model_file(country))

    metrics = {
        "country": country,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_records": len(processed),
        "mae": round(mae, 3),
        "r2_score": round(r2, 4),
    }
    _append_training_log(metrics)
    return metrics


def _append_training_log(metrics: dict):
    row = pd.DataFrame([metrics])
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(LOG_PATH, index=False)


def train_all_countries() -> list:
    """Train a model for every country CSV found in data/countries/."""
    results = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            country = fname.replace("_", " ").replace(".csv", "")
            # Recover original country names with spaces correctly for
            # multi-word countries by reading the Country column directly.
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            actual_country = df["Country"].iloc[0]
            metrics = train_country_model(actual_country, df)
            results.append(metrics)
            print(f"Trained {actual_country}: R2={metrics['r2_score']}, MAE={metrics['mae']}")
    return results


def load_model(country: str):
    path = _model_file(country)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found for '{country}'. Train it first.")
    return joblib.load(path)


if __name__ == "__main__":
    train_all_countries()
