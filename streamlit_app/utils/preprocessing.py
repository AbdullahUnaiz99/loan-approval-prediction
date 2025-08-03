import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_preprocessor():
    try:
        preprocessor = joblib.load("data/processed/preprocessor.joblib")
        return preprocessor
    except Exception:
        st.error("Preprocessor not found! Run preprocessing pipeline first.")
        return None

def create_input_form():
    st.subheader("Enter Loan Application Details")
    with st.form("loan_form"):
        loan_type = st.selectbox("Loan Type", ['Business Loan', 'Car Loan', 'Education Loan', 'Personal Loan', 'Home Loan'])
        loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1)
        interest_rate_offered = st.number_input("Interest Rate Offered (%)", min_value=0.0, format="%.2f")
        purpose_of_loan = st.selectbox("Purpose of Loan", ['Medical Emergency', 'Education', 'Debt Consolidation', 'Business Expansion', 'Wedding', 'Vehicle Purchase', 'Home Renovation'])
        employment_status = st.selectbox("Employment Status", ['Retired', 'Unemployed', 'Self-Employed', 'Salaried', 'Business Owner', 'Student'])
        monthly_income = st.number_input("Monthly Income", min_value=0)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
        existing_emis_monthly = st.number_input("Existing EMIs Monthly", min_value=0)
        debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=10.0, format="%.2f")
        property_ownership_status = st.selectbox("Property Ownership Status", ['Rented', 'Owned', 'Jointly Owned'])
        applicant_age = st.number_input("Applicant Age", min_value=18, max_value=100)
        gender = st.selectbox("Gender", ['Female', 'Other', 'Male'])
        number_of_dependents = st.number_input("Number of Dependents", min_value=0)
        loan_amount_requested = st.number_input("Loan Amount Requested", min_value=0)

        submitted = st.form_submit_button("Predict Loan Approval")

        if submitted:
            return {
                "loan_type": loan_type,
                "loan_tenure_months": loan_tenure_months,
                "interest_rate_offered": interest_rate_offered,
                "purpose_of_loan": purpose_of_loan,
                "employment_status": employment_status,
                "monthly_income": monthly_income,
                "cibil_score": cibil_score,
                "existing_emis_monthly": existing_emis_monthly,
                "debt_to_income_ratio": debt_to_income_ratio,
                "property_ownership_status": property_ownership_status,
                "applicant_age": applicant_age,
                "gender": gender,
                "number_of_dependents": number_of_dependents,
                "loan_amount_requested": loan_amount_requested,
            }
    return None

def preprocess_input(input_data, preprocessor):
    df = pd.DataFrame([input_data])
    processed_df = preprocessor.transform_new_data(df)
    return processed_df

def display_prediction_result(prediction, probability):
    if prediction == 1:
        st.success(f"Loan Approved with confidence {probability:.2%}")
    else:
        st.error(f"Loan Declined with confidence {(1-probability):.2%}")
