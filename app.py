import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.predict import predict_single

if not os.path.exists("models/best_model.pkl"):
    from src.train import main as train_main
    train_main()

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)
st.title("🏦 Loan Approval Predictor")
st.markdown("Fill in the applicant details below to predict loan approval.")
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male","Female"])
    married = st.selectbox("Married", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
    education = st.selectbox("Education", ["Graduate","Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes","No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    st.subheader("Income Details")
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=10000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=10000, step=500)

with col3:
    st.subheader("Loan Details")
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=150, step=10)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 24, 12])
    credit_history = st.selectbox("Credit History", [1, 0],
        format_func=lambda x: "Good (1)" if x == 1 else "Bad(0)")

st.divider()

if st.button("Predict", use_container_width=True):
    applicant = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    result = predict_single(applicant)
    st.divider()
    if result["approved"]:
        st.success(f"LOAN APPROVED - {result['probability']*100:.1f}% confident", icon="✓")
    else:
        st.error(f"LOAN REJECTED - {result['probability']*100:.1f}% confident", icon="✗")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Decision", "Approved ✅" if result["approved"] else "Rejected ❌")
    col_b.metric("Confidence", f"{result['probability']*100:.1f}%")
    col_c.metric("Risk Level", result["risk_level"])
    st.caption(f"Model used: {result['model_name']}")