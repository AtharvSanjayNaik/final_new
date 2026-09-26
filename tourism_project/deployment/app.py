"""
Streamlit web app for the "Visit with Us" Wellness Tourism Package
purchase-prediction model.

STATIC / OFFLINE MODE
----------------------
This app makes NO live Hugging Face Hub / Space SDK calls, and requires
no API key, token, payment, credits or subscription of any kind:

1. Primary path: load the trained model bundled locally with the app
   (best_tourism_model_v1.joblib), which was produced entirely offline
   by train.py. This is a fully local, static artifact - no network
   request of any kind is made to use it.
2. Fallback path: if that file is ever missing or fails to load, the
   app falls back to a deterministic, rule-based "static mock"
   predictor that mirrors the same business signal the real model
   learned (passport ownership, pitch engagement, income, product
   tier) so the feature keeps working and keeps showing realistic,
   project-specific results - never an error, never a payment/API-key
   prompt.

The active mode is always shown directly in the app UI.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


def static_mock_predict(row: dict):
    """
    Deterministic, rule-based stand-in for the trained model. Used only
    if the bundled local model artifact is unavailable. Mirrors the same
    business signal the real model learned from, so it keeps producing
    realistic, project-specific results - entirely offline, with no
    external Hugging Face Hub / Space SDK calls, API key, or payment of
    any kind.
    """
    score = 0.15
    if row.get("Passport") == 1:
        score += 0.25
    if row.get("PitchSatisfactionScore", 0) >= 4:
        score += 0.15
    if row.get("NumberOfFollowups", 0) >= 4:
        score += 0.10
    if row.get("MonthlyIncome", 999999) < 25000:
        score += 0.10
    if row.get("ProductPitched") in ("Basic", "Standard"):
        score += 0.10
    if row.get("Designation") == "Executive":
        score += 0.05
    score = float(min(max(score, 0.02), 0.97))
    prediction = 1 if score >= 0.5 else 0
    return prediction, score


@st.cache_resource
def load_model():
    """
    Returns (mode, model):
      mode == "local_model" -> model is a fitted sklearn Pipeline
      mode == "static_mock" -> model is None (static_mock_predict is used instead)
    No network calls are made here at all.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            return "local_model", joblib.load(LOCAL_MODEL_PATH)
        except Exception:
            pass
    return "static_mock", None


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="\U0001F9F3")
st.title("\U0001F9F3 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Predict whether a customer is likely to purchase the new **Wellness "
    "Tourism Package**, based on their profile and their interaction with "
    "the sales pitch."
)

mode, model = load_model()
if mode == "local_model":
    st.success(
        "\U0001F7E2 Prediction engine: **bundled local model** (fully offline - no "
        "external API calls, no Hugging Face Hub request, no payment or "
        "credentials required)."
    )
else:
    st.warning(
        "\U0001F7E1 Prediction engine: **static/mock mode** (no trained model artifact "
        "found - showing predefined, project-specific rule-based results. "
        "No external API calls, no payment, credits, subscription or API key "
        "are required or used)."
    )

st.header("Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (< age 5)", 0, 5, 0)
    NumberOfTrips = st.number_input("Avg Number of Trips per Year", 0, 20, 2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
    Passport = st.selectbox("Holds Valid Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

with col3:
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.number_input("Monthly Income", 1000, 100000, 22000, step=500)
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 10)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict Purchase Likelihood", type="primary"):
    input_dict = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }

    if mode == "local_model":
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    else:
        prediction, probability = static_mock_predict(input_dict)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"\u2705 Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"\u274C Unlikely to purchase the Wellness Package (probability: {probability:.1%})")

    st.caption(
        "Result computed by the "
        + ("bundled local model." if mode == "local_model" else "static/mock rule-based engine.")
        + " No external API or paid Space SDK request was made to produce this result."
    )
