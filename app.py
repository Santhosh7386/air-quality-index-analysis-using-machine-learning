"""
app.py
------
Country-Specific AQI Prediction System — Streamlit Application.

Master's Dissertation Project.

This is the main entry point. It wires together:
  - src/preprocessing.py  (data cleaning + feature engineering)
  - src/train.py          (per-country Gradient Boosting model training)
  - src/recommendations.py(AQI -> health advice)
  - src/aqi_utils.py       (AQI value -> category/color mapping)

Pages: Home, Predict AQI, Dashboard, Admin Panel, About.
Navigation is handled with a sidebar radio + st.session_state, which keeps
the whole app in a single file for clarity while still being fully modular
internally (each page is its own function).
"""

import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    clean_and_engineer, get_feature_columns, validate_uploaded_data,
    NUMERIC_FEATURES, SEASONS, TARGET,
)
from src.train import train_country_model, load_model, load_country_data, MODEL_DIR, LOG_PATH
from src.recommendations import get_recommendations
from src.aqi_utils import value_to_category, category_color, category_severity, AQI_BREAKPOINTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "countries")

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Global AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS — gives the app a professional analytics-dashboard look
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #f5f7fa;
    }

    /* Hero header */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.9;
    }

    /* Generic card */
    .dash-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eef0f3;
        margin-bottom: 1rem;
        height: 100%;
    }
    .dash-card h3 {
        margin-top: 0;
        font-size: 1.05rem;
        color: #2c3e50;
    }
    .dash-card p {
        color: #5a6b7b;
        font-size: 0.92rem;
    }

    /* AQI result card */
    .aqi-result-card {
        border-radius: 18px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    }
    .aqi-result-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
    }
    .aqi-result-label {
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    .rec-box {
        background: white;
        border-left: 5px solid #2c5364;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.3rem;
        border: none;
    }

    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f2f5 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.0rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def get_available_countries():
    countries = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, fname), nrows=1)
            countries.append(df["Country"].iloc[0])
    return sorted(countries)


def load_all_data():
    frames = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(DATA_DIR, fname)))
    return pd.concat(frames, ignore_index=True)


def model_exists(country):
    safe = country.replace(" ", "_")
    return os.path.exists(os.path.join(MODEL_DIR, f"{safe}_model.pkl"))


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
NAV_OPTIONS = ["🏠 Home", "📈 Predict AQI", "📊 Dashboard", "🛠 Admin Panel", "ℹ About"]

if "page" not in st.session_state:
    st.session_state.page = NAV_OPTIONS[0]

with st.sidebar:
    st.markdown("## 🌍 AQI Predictor")
    st.caption("Country-Specific Air Quality Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state.page),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown("---")
    st.caption("Master's Dissertation Project")
    st.caption(f"Countries supported: {len(get_available_countries())}")

st.session_state.page = page

# --------------------------------------------------------------------------
# PAGE: HOME
# --------------------------------------------------------------------------
def render_home():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌍 Country-Specific AQI Prediction System</h1>
            <p>Predicting Air Quality Index using per-country Gradient Boosting models,
            with real-time health recommendations for 20 countries worldwide.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_data = load_all_data()
    countries = get_available_countries()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.metric("🌎 Countries Covered", len(countries))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.metric("📊 Total Records", f"{len(all_data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.metric("🏙 Cities Tracked", all_data["City"].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        avg_aqi = round(all_data["AQI_Value"].mean(), 1)
        st.metric("🌡 Global Avg AQI", avg_aqi)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🗺 Average AQI by Country")
        country_avg = all_data.groupby("Country")["AQI_Value"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            country_avg, x="AQI_Value", y="Country", orientation="h",
            color="AQI_Value", color_continuous_scale=["#00A65A", "#F4C542", "#E9573F", "#6D0000"],
            labels={"AQI_Value": "Average AQI"}, height=520,
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🌡 AQI Category Distribution")
        cat_counts = all_data["AQI_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        color_map = {label: color for _l, _h, label, color, _r in AQI_BREAKPOINTS}
        fig2 = px.pie(
            cat_counts, names="Category", values="Count", hole=0.5,
            color="Category", color_discrete_map=color_map, height=520,
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🚀 How This System Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="dash-card">
            <h3>1️⃣ Select a Country</h3>
            <p>Choose a country and season, and enter current pollutant readings
            (PM2.5, PM10, NO2, SO2, CO, O3) on the Predict AQI page.</p>
            </div>""", unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="dash-card">
            <h3>2️⃣ Country-Specific Model Predicts</h3>
            <p>A Gradient Boosting Regressor trained specifically on that
            country's historical air quality data predicts the AQI value.</p>
            </div>""", unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="dash-card">
            <h3>3️⃣ Get Health Guidance</h3>
            <p>The system instantly displays the AQI category, health risk,
            and personalised recommendations and preventive measures.</p>
            </div>""", unsafe_allow_html=True,
        )

    st.write("")
    if st.button("📈 Go to Predict AQI →", type="primary"):
        st.session_state.page = "📈 Predict AQI"
        st.rerun()

# --------------------------------------------------------------------------
# PAGE: PREDICT AQI
# --------------------------------------------------------------------------
def render_predict():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📈 Predict Air Quality Index</h1>
            <p>Select a country and enter live pollutant readings to get an instant AQI prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    countries = get_available_countries()

    left, right = st.columns([1, 1.3])

    with left:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Input Parameters")

        country = st.selectbox("🌍 Country", countries)

        if not model_exists(country):
            st.warning(f"No trained model found for {country} yet. Please train it from the Admin Panel.")

        season = st.selectbox("🍂 Season", SEASONS)

        st.markdown("##### Pollutant Concentrations")
        c1, c2 = st.columns(2)
        with c1:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=600.0, value=50.0, step=1.0)
            no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=400.0, value=40.0, step=1.0)
            co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        with c2:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=800.0, value=90.0, step=1.0)
            so2 = st.number_input("SO2 (ppb)", min_value=0.0, max_value=300.0, value=20.0, step=1.0)
            o3 = st.number_input("O3 (ppb)", min_value=0.0, max_value=400.0, value=55.0, step=1.0)

        predict_btn = st.button("🔍 Predict AQI", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if predict_btn:
            if not model_exists(country):
                st.error(f"Cannot predict — no trained model exists for {country}. Train it in the Admin Panel first.")
            else:
                with st.spinner(f"Running {country}'s Gradient Boosting model..."):
                    bundle = load_model(country)
                    model = bundle["model"]
                    feature_cols = bundle["feature_columns"]

                    input_row = {
                        "PM2.5": pm25, "PM10": pm10, "NO2": no2,
                        "SO2": so2, "CO": co, "O3": o3,
                    }
                    for s in SEASONS:
                        input_row[f"Season_{s}"] = 1 if s == season else 0

                    X_input = pd.DataFrame([input_row])[feature_cols]
                    pred_aqi = float(model.predict(X_input)[0])
                    pred_aqi = max(0.0, pred_aqi)
                    category = value_to_category(pred_aqi)
                    color = category_color(category)
                    severity = category_severity(category)

                st.toast(f"Prediction complete for {country}!", icon="✅")

                st.markdown(
                    f"""
                    <div class="aqi-result-card" style="background: {color};">
                        <p style="opacity:0.85; margin:0; font-size:1rem;">Predicted AQI for {country} ({season})</p>
                        <p class="aqi-result-value">{pred_aqi:.1f}</p>
                        <p class="aqi-result-label">{category}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### Severity Level")
                st.progress(min(severity / 5, 1.0))

                rec = get_recommendations(category)

                tab1, tab2, tab3 = st.tabs(["⚠ Health Risk", "✅ Recommendations", "🛡 Preventive Measures"])
                with tab1:
                    st.markdown(f'<div class="rec-box"><b>Most affected:</b> {rec["most_affected"]}<br><br>{rec["health_risk"]}</div>', unsafe_allow_html=True)
                with tab2:
                    for r in rec["recommendations"]:
                        st.markdown(f'<div class="rec-box">✅ {r}</div>', unsafe_allow_html=True)
                with tab3:
                    for p in rec["preventive_measures"]:
                        st.markdown(f'<div class="rec-box">🛡 {p}</div>', unsafe_allow_html=True)

                with st.expander("📊 View Input Feature Breakdown"):
                    input_df = pd.DataFrame([input_row]).T.reset_index()
                    input_df.columns = ["Feature", "Value"]
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """<div class="dash-card" style="text-align:center; padding:3rem;">
                <h3>👈 Enter pollutant readings and click Predict AQI</h3>
                <p>Your predicted AQI, category, health risk and recommendations will appear here.</p>
                </div>""",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------------------------------
def render_dashboard():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📊 Air Quality Analytics Dashboard</h1>
            <p>Explore historical trends, pollutant relationships, and model performance across countries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_data = load_all_data()
    countries = get_available_countries()

    filt_col1, filt_col2 = st.columns([1, 1])
    with filt_col1:
        selected_countries = st.multiselect("Filter by country", countries, default=countries[:5])
    with filt_col2:
        selected_seasons = st.multiselect("Filter by season", SEASONS, default=SEASONS)

    filtered = all_data[
        all_data["Country"].isin(selected_countries if selected_countries else countries)
        & all_data["Season"].isin(selected_seasons if selected_seasons else SEASONS)
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records in view", f"{len(filtered):,}")
    k2.metric("Avg AQI", round(filtered["AQI_Value"].mean(), 1) if len(filtered) else 0)
    k3.metric("Max AQI", round(filtered["AQI_Value"].max(), 1) if len(filtered) else 0)
    k4.metric("Min AQI", round(filtered["AQI_Value"].min(), 1) if len(filtered) else 0)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🧪 Pollutant Correlation", "🌦 Seasonal Patterns", "🎯 Model Performance"])

    with tab1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("#### AQI Distribution by Country")
        if len(filtered):
            fig = px.box(filtered, x="Country", y="AQI_Value", color="Country", points=False, height=460)
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one country and season to see data.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("#### Pollutant Correlation Heatmap")
        if len(filtered):
            corr = filtered[NUMERIC_FEATURES + ["AQI_Value"]].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", height=460, aspect="auto")
            fig.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("PM2.5 and PM10 typically show the strongest correlation with AQI_Value across most countries.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("#### Average AQI by Season")
        if len(filtered):
            season_avg = filtered.groupby("Season")["AQI_Value"].mean().reindex(SEASONS).reset_index()
            fig = px.bar(season_avg, x="Season", y="AQI_Value", color="Season", height=420)
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("#### Per-Country Model Performance (R² and MAE)")
        if os.path.exists(LOG_PATH):
            log_df = pd.read_csv(LOG_PATH)
            latest = log_df.sort_values("timestamp").groupby("country").tail(1).sort_values("r2_score", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=latest["country"], y=latest["r2_score"], name="R² Score", marker_color="#2c5364"))
            fig.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="R² Score", xaxis_title="Country",
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 View raw training log"):
                st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No training log found yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PAGE: ADMIN PANEL
# --------------------------------------------------------------------------
def render_admin():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🛠 Admin Panel</h1>
            <p>Upload new records for a country, validate them, merge with existing data, and retrain
            only that country's model — without touching any other country's model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    countries = get_available_countries()

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 1️⃣ Select Country")
        admin_country = st.selectbox("Country to update", countries, key="admin_country")

        existing_df = load_country_data(admin_country)
        st.metric("Current records for this country", len(existing_df))

        if os.path.exists(LOG_PATH):
            log_df = pd.read_csv(LOG_PATH)
            country_log = log_df[log_df["country"] == admin_country].sort_values("timestamp")
            if len(country_log):
                last = country_log.iloc[-1]
                st.caption(f"Last trained: {last['timestamp']}  |  R²={last['r2_score']}  |  MAE={last['mae']}")
            else:
                st.caption("This country has not been trained yet.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 2️⃣ Upload New Records")
        st.caption("CSV must contain columns: Country, City, Season, PM2.5, PM10, NO2, SO2, CO, O3, AQI_Value, AQI_Category")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="admin_upload")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 3️⃣ Validate, Merge & Retrain")

        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the uploaded file: {e}")
                new_df = None

            if new_df is not None:
                is_valid, message = validate_uploaded_data(new_df)

                if not is_valid:
                    st.error(f"❌ Validation failed: {message}")
                else:
                    st.success(f"✅ {message}  ({len(new_df)} rows detected)")

                    mismatched = new_df[new_df["Country"] != admin_country]
                    if len(mismatched):
                        st.warning(
                            f"⚠ {len(mismatched)} row(s) have a Country value different from "
                            f"'{admin_country}' and will be reassigned to '{admin_country}' before merging, "
                            f"since this upload targets that country only."
                        )
                        new_df = new_df.copy()
                        new_df["Country"] = admin_country

                    st.markdown("##### Preview of uploaded data")
                    st.dataframe(new_df.head(10), use_container_width=True, hide_index=True)

                    st.markdown("##### Merge Summary")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Existing records", len(existing_df))
                    m2.metric("New records", len(new_df))
                    m3.metric("Total after merge", len(existing_df) + len(new_df))

                    if st.button(f"🔁 Merge & Retrain {admin_country} Model", type="primary", use_container_width=True):
                        with st.spinner(f"Merging data and retraining Gradient Boosting model for {admin_country}..."):
                            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                            merged_df = merged_df.drop_duplicates()
                            metrics = train_country_model(admin_country, merged_df)

                        st.toast(f"{admin_country} model retrained successfully!", icon="🎉")
                        st.success(
                            f"✅ Retraining complete for **{admin_country}**.\n\n"
                            f"- Total records used: {metrics['n_records']}\n"
                            f"- R² Score: {metrics['r2_score']}\n"
                            f"- MAE: {metrics['mae']}\n\n"
                            f"Other countries' models were **not** modified."
                        )
                        st.balloons()
        else:
            st.info("Upload a CSV file above to begin the validation and retraining workflow.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Full Training Log")
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No training has occurred yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PAGE: ABOUT
# --------------------------------------------------------------------------
def render_about():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>ℹ About This Project</h1>
            <p>Country-Specific AQI Prediction System — Master's Dissertation Project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Objective")
        st.write(
            "This system predicts the Air Quality Index (AQI) for 20 countries using "
            "country-specific Gradient Boosting Regressor models, rather than a single "
            "global model. Air pollution dynamics vary meaningfully between countries "
            "due to differing pollution sources, regulations, and climates — training a "
            "dedicated model per country captures these local patterns more accurately "
            "than a one-size-fits-all approach."
        )
        st.markdown("### 🧠 Machine Learning Approach")
        st.write(
            "AQI_Value is treated as a continuous target (regression). The AQI category "
            "(e.g. Good, Moderate, Hazardous) is derived deterministically from the "
            "predicted value using standard EPA-style breakpoints, guaranteeing the "
            "category always matches the numeric prediction. Gradient Boosting was chosen "
            "because it models non-linear relationships between pollutants (PM2.5, PM10, "
            "NO2, SO2, CO, O3) and AQI effectively, is robust to outliers, and does not "
            "require feature scaling."
        )
        st.markdown("### 🏗 System Architecture")
        st.code(
            "aqi_app/\n"
            "├── app.py                 # Streamlit application (this file)\n"
            "├── data/countries/*.csv   # Per-country datasets\n"
            "├── models/*_model.pkl     # One trained model per country\n"
            "├── src/\n"
            "│   ├── preprocessing.py   # Cleaning + feature engineering\n"
            "│   ├── train.py           # Model training / retraining logic\n"
            "│   ├── recommendations.py # AQI -> health advice mapping\n"
            "│   └── aqi_utils.py       # AQI value -> category/color mapping\n"
            "└── logs/training_log.csv  # Full training history",
            language="text",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🛠 Tech Stack")
        for tech in ["Python", "Streamlit", "Pandas & NumPy", "Scikit-learn", "Joblib", "Plotly"]:
            st.markdown(f"- {tech}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 📐 AQI Category Scale")
        for low, high, label, color, _rank in AQI_BREAKPOINTS:
            st.markdown(
                f'<span class="metric-pill" style="background:{color};">{label} ({low:.0f}-{high:.0f})</span><br><br>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown("### 🔒 Admin Workflow Guarantee")
        st.write(
            "Retraining a country's model only ever affects that country's dataset "
            "and model file. All other countries' models remain untouched, which is "
            "verified by scoping every file operation to a single country's CSV and "
            "PKL file path."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
if st.session_state.page == "🏠 Home":
    render_home()
elif st.session_state.page == "📈 Predict AQI":
    render_predict()
elif st.session_state.page == "📊 Dashboard":
    render_dashboard()
elif st.session_state.page == "🛠 Admin Panel":
    render_admin()
elif st.session_state.page == "ℹ About":
    render_about()
