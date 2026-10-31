"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(page_title="Wellness Package Purchase Predictor", page_icon="🧳", layout="wide")


@st.cache_resource
def load_model():
    """Load the model committed to the repository by the GitHub Actions pipeline."""
    return joblib.load(MODEL_PATH)


st.title("🧳 Visit with Us - Wellness Tourism Package Predictor")
st.write(
    "Enter the customer's details to predict whether they are likely to purchase the "
    "**Wellness Tourism Package** before the sales team contacts them."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline to train and commit the model.")
    st.stop()
model = load_model()

# ------------------------------------------------------------------ Inputs
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"], key="marital")
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], key="occupation")
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], key="designation")
with c2:
    monthly_income = st.number_input("Monthly income", min_value=0, max_value=200000, value=22000, step=500, key="income")
    city_tier = st.selectbox("City tier", [1, 2, 3], key="city_tier")
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"], index=1, key="passport")
    own_car = st.selectbox("Owns a car?", ["Yes", "No"], key="own_car")
    number_of_trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3, key="trips")
with c3:
    persons_visiting = st.number_input("People on the trip", min_value=1, max_value=10, value=3, key="persons")
    children_visiting = st.number_input("Children under 5 on the trip", min_value=0, max_value=5, value=1, key="children")
    property_star = st.selectbox("Preferred hotel rating", [3, 4, 5], key="property_star")

st.subheader("Sales interaction")
c4, c5, c6 = st.columns(3)
with c4:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"], key="contact")
    product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], key="product")
with c5:
    duration_of_pitch = st.number_input("Pitch duration (minutes)", min_value=1, max_value=180, value=15, key="duration")
    number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4, key="followups")
with c6:
    pitch_satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3, key="satisfaction")

# Save the inputs into a dataframe with the same columns (and order) used in training
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Model input (dataframe)"):
    st.dataframe(input_df, hide_index=True)

# ------------------------------------------------------------------ Prediction
if st.button("Predict purchase", type="primary", key="predict"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Tourism Package (probability {probability:.1%}).")
        st.write("Recommendation: prioritise this customer for a personalised pitch and timely follow-ups.")
    else:
        st.warning(f"❌ Unlikely to purchase the Wellness Tourism Package (probability {probability:.1%}).")
        st.write("Recommendation: keep in nurture campaigns rather than an immediate sales call.")

if METADATA_PATH.exists():
    meta = json.loads(METADATA_PATH.read_text())
    st.caption(
        f"Model trained {meta['trained_at_utc']} UTC | test F1 {meta['test_metrics']['f1']:.3f} | "
        f"test recall {meta['test_metrics']['recall']:.3f} | test ROC-AUC {meta['test_metrics']['roc_auc']:.3f}"
    )
