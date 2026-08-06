# 🌍 Country-Specific AQI Prediction System

A full-stack Machine Learning + Streamlit web application built as a Master's
Dissertation project. It predicts Air Quality Index (AQI) using a dedicated
**Gradient Boosting Regressor trained per country** (20 countries), and
provides real-time health recommendations based on the predicted AQI.

## 1. Project Summary

| Item | Detail |
|---|---|
| Dataset | `global_air_quality_dataset.csv` — 2,000 records, 20 countries, 6 pollutants |
| Target | `AQI_Value` (continuous, regression problem) |
| Category | `AQI_Category` — deterministically derived from `AQI_Value` via EPA-style breakpoints |
| Model | One `GradientBoostingRegressor` per country (20 models total) |
| Features | PM2.5, PM10, NO2, SO2, CO, O3 + one-hot encoded Season |
| Framework | Streamlit (frontend) + Scikit-learn (ML backend) |

## 2. Why This Approach?

- **Regression, not classification**: `AQI_Category` is fully determined by
  `AQI_Value` via fixed breakpoints (verified during EDA — every category has
  a clean, non-overlapping AQI_Value range). So the model predicts the
  numeric AQI, and the category/health-risk/recommendations are derived
  afterwards from that number. This avoids class-imbalance issues entirely.
- **Per-country models**: Pollution sources, regulation, and typical AQI
  ranges differ substantially between countries. A single global model
  would blur these country-specific relationships; per-country models let
  each one specialise (visible in the varying R² scores per country on the
  Dashboard page).
- **Gradient Boosting Regressor**: Handles non-linear relationships between
  pollutants and AQI, is robust to the outliers present in raw pollutant
  readings, does not require feature scaling, and gives strong performance
  even on the small per-country subsets (~80-115 rows each) used here.
- **Median imputation**: PM2.5, PM10, NO2, and SO2 each had ~100 missing
  values (5%) in the raw dataset. Median imputation was used because it is
  robust to the right-skewed, outlier-heavy distribution of pollutant data.

## 3. Folder Structure

```
aqi_app/
├── app.py                     # Streamlit application (Home, Predict, Dashboard, Admin, About)
├── requirements.txt
├── data/
│   └── countries/*.csv        # One CSV per country (split from the original dataset)
├── models/
│   └── {Country}_model.pkl    # One trained Gradient Boosting model per country
├── src/
│   ├── preprocessing.py       # Validation + cleaning + feature engineering (shared)
│   ├── train.py                # Training / retraining logic
│   ├── recommendations.py      # AQI category -> health risk / recommendations / preventive measures
│   └── aqi_utils.py            # AQI value -> category / color / severity mapping
└── logs/
    └── training_log.csv        # Full history of every training/retraining run
```

## 4. How to Run

```bash
cd aqi_app
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser (default: http://localhost:8501).
All 20 country models are **already trained and included** — no setup step
is required before you can start predicting.

## 5. Retraining a Model Yourself

To retrain all 20 country models from scratch:

```bash
python -m src.train
```

To retrain a single country's model with new data, use the **Admin Panel**
page inside the running app: select a country, upload a CSV with the
required columns, and click "Merge & Retrain". Only that country's model
and dataset are modified.

## 6. App Pages

- **🏠 Home** — Project overview, global KPIs, average AQI by country, category distribution.
- **📈 Predict AQI** — Select country + season, enter pollutant readings, get instant AQI prediction, category, health risk, recommendations, and preventive measures.
- **📊 Dashboard** — Filterable trend charts, pollutant correlation heatmap, seasonal patterns, and per-country model performance (R²/MAE).
- **🛠 Admin Panel** — Upload new records for a specific country, validate them, merge with existing data, and retrain only that country's model.
- **ℹ About** — Project objective, ML approach, architecture, and AQI scale reference.

## 7. Model Performance Summary (initial training)

See `logs/training_log.csv` for the full, continuously-updated history. R²
ranges roughly from 0.42 (smallest datasets, e.g. Canada/Australia at ~85-100
rows) to 0.96 (India), reflecting the well-known relationship between
training set size and Gradient Boosting performance — a good discussion
point for the dissertation's limitations section.

## 8. Key Dissertation Talking Points

1. Deterministic category derivation (regression + fixed breakpoints) rather
   than direct classification — avoids information loss and class imbalance.
2. Per-country modelling as a deliberate architectural choice, with
   quantitative evidence (varying R²) that pollutant-AQI relationships
   genuinely differ by country.
3. A reusable, testable recommendation module decoupled from both the ML
   code and the UI code (separation of concerns).
4. A fully functional Admin/retraining workflow demonstrating an MLOps-lite
   loop: upload → validate → merge → retrain → log — scoped safely to a
   single country at a time.
