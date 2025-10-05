import os
import joblib
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# ================================
# App Title & Description
# ================================
st.set_page_config(page_title="Tourism Package Prediction", page_icon="🌍", layout="centered")

st.title("🌍 Tourism Package Prediction App")
st.markdown(
    """
    Provide customer details below to predict whether they are likely to
    **opt for the Wellness Tourism Package**.
    """
)
 

# ================================
# Load Model from Hugging Face Hub
# ================================
@st.cache_resource(show_spinner=True)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="cbendale10/MLOps-Tourism-Prediction-model",
        filename="best_machine_failure_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# ================================
# Input Form (kept exactly as requested)
# ================================
with st.form("input_form"):
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=35)
        CityTier = st.selectbox("CityTier", options=[1, 2, 3], index=0)
        NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=20, value=2)
        PreferredPropertyStar = st.selectbox("PreferredPropertyStar", options=[1, 2, 3, 4, 5], index=2)
        NumberOfTrips = st.number_input("NumberOfTrips (per year)", min_value=0, max_value=50, value=2)
        NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (under 5)", min_value=0, max_value=10, value=0)

    with col2:
        TypeofContact = st.selectbox("TypeofContact", options=["Company Invited", "Self Enquiry"], index=0)
        Occupation = st.selectbox("Occupation", options=["Salaried", "Freelancer"], index=0)
        Gender = st.selectbox("Gender", options=["Male", "Female"], index=1)
        MaritalStatus = st.selectbox("MaritalStatus", options=["Single", "Married", "Divorced"], index=1)
        Designation = st.selectbox("Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)
        ProductPitched = st.selectbox("ProductPitched", options=["Basic", "Deluxe", "Premium", "Super Deluxe", "King"], index=1)

    st.subheader("Interaction Details")
    col3, col4 = st.columns(2)
    with col3:
        Passport = st.selectbox("Passport", options=[0, 1], index=1)
        OwnCar = st.selectbox("OwnCar", options=[0, 1], index=0)
        PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", options=[1, 2, 3, 4, 5], index=2)
    with col4:
        NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=3)
        DurationOfPitch = st.number_input("DurationOfPitch (minutes)", min_value=0, max_value=120, value=10)
        MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=10_000_000, value=25_000, step=100)

    st.caption("Note: All inputs should mirror the training features your model expects.")
    submitted = st.form_submit_button("Predict 💡")

if submitted:
    # Build a single-row DataFrame payload
    payload = {
        "Age": Age,
        "CityTier": CityTier,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation,
        "ProductPitched": ProductPitched,
    }

    st.subheader("Input preview")
    st.dataframe(pd.DataFrame([payload]))

    CLASSIFICATION_THRESHOLD = 0.45

     
    try:
        df = pd.DataFrame([payload])
        proba = float(model.predict_proba(df)[0][1])
        st.success(f"Predicted purchase probability: **{proba:.3f}**")
        decision = "Likely to Purchase ✅" if proba >= CLASSIFICATION_THRESHOLD else "Unlikely to Purchase ❌"
        st.metric("Decision", decision)
        st.caption(f"Threshold = {CLASSIFICATION_THRESHOLD:.2f} (override via CLASSIFICATION_THRESHOLD env var)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.markdown("---")
 
