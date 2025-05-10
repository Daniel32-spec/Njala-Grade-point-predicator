
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained model (for demo purposes, we retrain here — in production use joblib.load)
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

st.markdown("This app predicts whether a student has passed or failed for an academic year based on GPA, credit hours, and year average.")

# User inputs
gpa = st.slider("GPA", min_value=1.0, max_value=4.0, step=0.01, value=2.5)
credit_hours = st.selectbox("Credit Hours", [3, 4, 6])
year_avg = st.slider("Year Average", min_value=1.0, max_value=4.0, step=0.01, value=2.5)

if st.button("Predict Result"):
    prediction = model.predict([[gpa, credit_hours, year_avg]])[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.success(f"The predicted result is: {result}")
