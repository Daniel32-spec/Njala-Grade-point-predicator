
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load and train model on start (demo purpose)
@st.cache_resource
def train_model():
    df = pd.read_csv('njala_student_data.csv')
    df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})
    X = df[['GPA', 'Credit_Hours', 'Year_Average']]
    y = df['Result']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

st.title("🎓 Njala University Pass/Fail Predictor")

st.markdown("This app predicts whether a student has passed or failed an academic year based on GPA, credit hours, and year average.")

# New Inputs: Module name and specific credit hours
module_name = st.text_input("📘 Enter Module Name", placeholder="e.g., MATH202")
module_credit = st.number_input("📐 Enter Credit Hours for This Module", min_value=1, max_value=6, step=1, value=3)

# Existing Inputs
gpa = st.slider("GPA", min_value=1.0, max_value=4.0, step=0.01, value=2.5)
year_avg = st.slider("Year Average", min_value=1.0, max_value=4.0, step=0.01, value=2.5)

if st.button("Predict Result"):
    prediction = model.predict([[gpa, module_credit, year_avg]])[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.success(f"Module: {module_name or 'N/A'} | Predicted Result: {result}")
