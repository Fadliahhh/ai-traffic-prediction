import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AI Traffic Prediction",
    page_icon="🚦",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "traffic_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "File traffic_model.joblib tidak ditemukan. "
            "Pastikan file model sudah ada di repository."
        )

    artifact = joblib.load(MODEL_PATH)
    return artifact

try:
    artifact = load_model()
    model = artifact["model"]
    features = artifact["features"]
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================
# HEADER
# =========================
st.title("🚦 AI Traffic Prediction")

st.write(
    "Predict traffic volume using historical traffic "
    "and weather patterns."
)

st.caption(
    "Machine learning model: Random Forest Regressor"
)

st.divider()

# =========================
# USER INPUT
# =========================
st.subheader("Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Date",
        value=date.today()
    )

    hour = st.slider(
        "Hour of the day",
        min_value=0,
        max_value=23,
        value=8
    )

    temp_c = st.number_input(
        "Temperature (°C)",
        min_value=-50.0,
        max_value=60.0,
        value=20.0,
        step=1.0
    )

    rain = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=0.1
    )

with col2:
    snow = st.number_input(
        "Snowfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=0.1
    )

    clouds = st.slider(
        "Cloud cover (%)",
        min_value=0,
        max_value=100,
        value=40
    )

    weather = st.selectbox(
        "Weather condition",
        [
            "Clear",
            "Clouds",
            "Rain",
            "Drizzle",
            "Mist",
            "Fog",
            "Haze",
            "Snow",
            "Thunderstorm",
            "Smoke"
        ]
    )

    holiday = st.selectbox(
        "Holiday",
        ["None", "Holiday"]
    )

# =========================
# PREPARE INPUT
# =========================
if st.button("Predict Traffic", type="primary",
             use_container_width=True):

    day_of_week = selected_date.weekday()
    month = selected_date.month
    is_weekend = int(day_of_week >= 5)

    # Dataset temperature uses Kelvin
    temp_kelvin = temp_c + 273.15

    input_data = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "temp": temp_kelvin,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "holiday": holiday,
        "weather_main": weather
    }])

    # Ensure the input columns match training
    input_data = input_data[features]

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Traffic Volume",
        value=f"{prediction:,.0f} vehicles"
    )

    if prediction < 1000:
        st.success("Relatively low traffic")
    elif prediction < 3000:
        st.warning("Moderate traffic")
    else:
        st.error("Relatively high traffic")

    st.caption(
        "This is a model-based estimate, not live traffic data. "
        "The model was trained on historical traffic from "
        "the I-94 region in the United States."
    )

# =========================
# FOOTER
# =========================
st.divider()

st.caption(
    "AI Traffic Prediction | Educational STEAM Project"
)
