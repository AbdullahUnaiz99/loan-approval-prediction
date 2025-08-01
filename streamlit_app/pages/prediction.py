import streamlit as st
import pandas as pd
import joblib

def render():
    st.header("🔮 Make a Prediction")
    model = joblib.load("models/best_model.joblib")

    st.subheader("Input Features")
    credit_history = st.selectbox("Credit History", [0, 1])
    applicant_income = st.number_input("Applicant Income", 0.0)
    loan_amount = st.number_input("Loan Amount", 0.0)

    if st.button("Predict"):
        X = pd.DataFrame(
            {
                "Credit_History": [credit_history],
                "ApplicantIncome": [applicant_income],
                "LoanAmount": [loan_amount],
            }
        )
        pred = model.predict(X)[0]
        st.success(f"Prediction: **{pred}**")
