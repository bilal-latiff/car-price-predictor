# Section 8: Interactive web application built with Streamlit
# Run this app from a terminal with:  streamlit run app.py
# (This cell writes the app.py file; it does not launch the app from inside the notebook.)

import streamlit as st
import pandas as pd
import joblib

# Load the best-performing model (Gradient Boosting) and the supporting encoder/feature list
# that were saved at the end of Section 7
model = joblib.load('best_model.pkl')
brand_encoder = joblib.load('brand_encoder.pkl')
final_features = joblib.load('final_features.pkl')

st.set_page_config(page_title="Car Price Predictor", page_icon="car")
st.title("Used Car Selling Price Predictor")
st.write("Enter the details of a used car below to estimate its selling price.")

# --- User inputs ---
vehicle_age = st.slider("Vehicle age (years)", 0, 20, 5)
km_driven = st.number_input("Kilometers driven", min_value=0, max_value=200000, value=50000, step=1000)
engine = st.number_input("Engine capacity (cc)", min_value=600, max_value=4000, value=1200, step=50)
max_power = st.number_input("Max power (bhp)", min_value=30.0, max_value=300.0, value=85.0, step=1.0)
seats = st.selectbox("Number of seats", [2, 4, 5, 6, 7, 8, 9], index=2)
fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Transmission type", ["Manual", "Automatic"])

# --- Build a single-row dataframe matching the model's expected input columns ---
input_dict = {
    'max_power': max_power,
    'vehicle_age': vehicle_age,
    'engine': engine,
    'fuel_type_Diesel': 1 if fuel_type == 'Diesel' else 0,
    'fuel_type_Petrol': 1 if fuel_type == 'Petrol' else 0,
    'seats': seats,
    'transmission_type_Manual': 1 if transmission_type == 'Manual' else 0,
    'km_driven': km_driven,
}
input_df = pd.DataFrame([input_dict])[final_features]

if st.button("Predict Selling Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Selling Price: Rs. {prediction:,.0f}")
    st.caption("This is an estimate based on a Gradient Boosting model trained on historical listings "
               "and should be used as a guide only, not a formal valuation.")
