import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/credit_model.pkl")

st.title("💳 Credit Risk Prediction System")

# Inputs
age = st.slider("Age", 18, 75)
credit_amount = st.number_input("Credit Amount")
duration = st.slider("Loan Duration (months)", 6, 60)

# Create dataframe
input_data = pd.DataFrame(
    [[age, credit_amount, duration]],
    columns=["Age", "Credit amount", "Duration"]
)

# Predict button
if st.button("Predict Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("### Probability of Default:", round(probability,2))

    if probability < 0.2:
        st.success("AAA - Low Risk")
    elif probability < 0.4:
        st.info("BBB - Medium Risk")
    else:
        st.error("D - High Risk")