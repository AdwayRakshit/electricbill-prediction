import streamlit as st
import pandas as pd
import pickle

# STEP 1: Configure the page
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="centered"
)

# STEP 2: Load the saved model (cached)
@st.cache_resource
def load_model():
    with open("electricity_model.pkl", "rb") as file:
        package = pickle.load(file)
    return package["model"], package["features"]

model, features = load_model()

# STEP 3: Heading and description
st.title("⚡ Smart Electricity Consumption Predictor")
st.write(
    "Estimate hourly electricity consumption "
    "using a trained Linear Regression model."
)
st.info(
    "Enter office conditions for a one-hour period "
    "to estimate electricity consumption."
)

# STEP 4: Collect office information
st.subheader("Office Conditions")

temperature_c = st.number_input(
    "Outdoor temperature (Celsius)",
    min_value=-5.0, max_value=50.0, value=34.0, step=1.0
)

people_count = st.number_input(
    "Number of people",
    min_value=0, max_value=100, value=12, step=1
)

ac_hours = st.number_input(
    "AC operating hours during this hour",
    min_value=0.0, max_value=1.0, value=1.0, step=0.1
)

computer_count = st.number_input(
    "Number of operating computers",
    min_value=0, max_value=100, value=8, step=1
)

lighting_hours = st.number_input(
    "Lighting operating hours during this hour",
    min_value=0.0, max_value=1.0, value=1.0, step=0.1
)

# STEP 5: Predict when the button is clicked
if st.button("Predict Electricity Consumption"):

    new_data = pd.DataFrame(
        [[
            temperature_c,
            people_count,
            ac_hours,
            computer_count,
            lighting_hours
        ]],
        columns=features
    )

    prediction = model.predict(new_data)[0]
    prediction = max(0, prediction)

    st.success(f"Estimated consumption: {prediction:.2f} kWh")
    st.caption(
        "This is an estimate for a one-hour period, "
        "not a measured electricity reading."
    )

st.divider()
st.caption(
    "Educational project | Python | Pandas | "
    "Scikit-learn | Pickle | Streamlit"
)