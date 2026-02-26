
# Importing required libraries to build a UI app
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Lokeshnathy/Best-tour-pur-pred-model",filename="best_tour_purchase_pred_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for tourism package purchse prediction
st.title("Wellness Tourism Package - Purchase Prediction")
st.write("""
This application predicts if a self enquired customer purchase the newly implemented **Wellness Tourism Package**.
This application is meant for the internal use of the company.
Kindly, fill below the required fields to get a prediction.
""")
# Collect customer details and company's interaction details with the customer
Contacted = st.selectbox("Contact Through",['Self Enquiry'])
Age = st.number_input("Age of the customer",min_value=10,max_value=70,value=30,step=1)
CityTier = st.selectbox("City tier customer belongs to:",['Tier 1','Tier 2','Tier 3'])
Occupation = st.selectbox("Occupation",['Small Business','Large Business','Salaried','Freelancer'])
Gender = st.selectbox("Select Gender",['Male','Female'])
MaritalStatus = st.selectbox("Marital Status",['Single','Unmarried','Married','Divorced'])
Passport = st.selectbox("Possess valid Passport",['Yes','No'])
OwnCar = st.selectbox("Does the customer has an own car ?",['Yes','No'])
Designation = st.selectbox("Designation - select relevant option",['Executive','Manager','VP','AVP','Senior Manager'])
MonthlyIncome = st.number_input("Monthly Income",min_value=0, max_value=100000,value=10000,step=1000)
Avg_Trips = st.number_input("Trip Count",min_value=0,max_value=25,value=1,step=1)
Accom_Person = st.number_input("Number of elders accompanying",min_value=0,max_value=10,value=1,step=1)
Accom_child = st.number_input("Number of children accompanying",min_value=0,max_value=10,value=1,step=1)
Pitch_Duration = st.number_input("Duration of Sales Pitch",min_value=0,max_value=100,value=10,step=1)
Followups = st.number_input("Number of Follow-Ups made",min_value=0,max_value=10,value=1,step=1)
ProductPitched = st.selectbox("Select the product",['Basic','Standard','Deluxe','Super Deluxe','King'])
PropertyRating= st.selectbox("Preferred Property Rating provided by customer",['3','4','5'])
SatisfactionScore = st.number_input("Sales pitch satisfaction level",min_value=1,max_value=5,value=5,step=1)

# Convert inputs into DataFrame
input_data = pd.DataFrame([{
    'Contacted': Contacted,
    'Age': Age,
    'CityTier': 1 if CityTier=="Tier 1" else 2 if CityTier=="Tier 2" else 3,
    'Occupation':Occupation,
    'Gender': Gender,
    'MaritalStatus':MaritalStatus,
    'Passport': 1 if Passport=="Yes" else 0,
    'OwnCar': 1 if OwnCar=="Yes" else 0,
    'Designation':Designation,
    'MonthlyIncome':MonthlyIncome,
    'Avg_Trips':Avg_Trips,
    'Accom_Person': Accom_Person,
    'Accom_child':Accom_child,
    'Pitch_Duration': Pitch_Duration,
    'Followups': Followups,
    'ProductPitched':ProductPitched,
    'PropertyRating':PropertyRating,
    'SatisfactionScore':SatisfactionScore}])
# Set the classification threshold
classification_threshold=0.45
# Predict Button
if st.button("Predict"):
  prediction_proba = model.predict_proba(input_data)[0,1]
  prediction = (prediction_proba>=classification_threshold).astype(int)
  result="purchase" if prediction == 1 else "not purchase"
  st.write(f"Based on the information provided, the potential customer is likely to {result} the Tourism Package.")
