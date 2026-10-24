
import streamlit as st
import pandas as pd
import joblib

from datetime import date

st.set_page_config(
    page_title="AI Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    return joblib.load("traffic_model.joblib")

artifact = load_artifact()
model = artifact["model"]

st.title("AI Traffic Prediction")
st.caption(
    "Estimate hourly traffic volume using historical "
    "traffic patterns and weather conditions."
)

st.info(
    "Research demo based on historical traffic data "
    "from Interstate 94, United States. "
    "Not a live traffic service."
)

with st.form("prediction_form"):
    st.subheader("Prediction inputs")

    selected_date = st.date_input(
        "Date",
        value=date(2018, 10, 1),
        min_value=date(2012, 1, 1),
        max_value=date(2035, 12, 31)
    )

    hour = st.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=8
    )

    temp_c = st.slider(
        "Temperature (°C)",
        min_value=-30,
        max_value=45,
        value=15
    )

    rain = st.number_input(
        "Rainfall (mm/hour)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    snow = st.number_input(
        "Snowfall (mm/hour)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    clouds = st.slider(
        "Cloud cover (%)",
        min_value=0,
        max_value=100,
        value=50
    )

    weather = st.selectbox(
        "Weather",
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
            "Smoke",
            "Squall"
        ]
    )

    holiday = st.text_input(
        "Holiday name (enter None if not a holiday)",
        value="None"
    )

    submitted = st.form_submit_button(
        "Predict traffic volume",
        type="primary"
    )

if submitted:
    input_data = pd.DataFrame([{
        "hour": hour,
        "day_of_week": selected_date.weekday(),
        "month": selected_date.month,
        "is_weekend": int(selected_date.weekday() >= 5),
        "temp": temp_c + 273.15,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "holiday": holiday.strip() or "None",
        "weather_main": weather
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, prediction)

    st.divider()
    st.subheader("Predicted hourly traffic volume")

    st.metric(
        label="Estimated vehicles per hour",
        value=f"{prediction:,.0f}"
    )

    st.caption(
        "This is a model estimate, not a live count "
        "or a guaranteed future traffic measurement."
    )
