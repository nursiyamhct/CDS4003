import streamlit as st
import pandas as pd
import joblib

st.title("Bank Loan Prediction — Teaching Demo")
st.caption("Synthetic educational model; not for real lending decisions.")

pipeline = joblib.load("pipeline.pkl")
income = st.number_input("Monthly income (AED)", min_value=0, value=12000)
loan = st.number_input("Loan amount (AED)", min_value=0, value=60000)
credit = st.number_input("Credit score", min_value=300, max_value=850, value=720)

if st.button("Predict"):
    customer = pd.DataFrame([{
        "monthly_income": income,
        "loan_amount": loan,
        "credit_score": credit
    }])
    prediction = pipeline.predict(customer)[0]
    st.write("Demo prediction:", "Approved" if prediction == 1 else "Not approved")
