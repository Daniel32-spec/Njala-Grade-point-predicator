
# 🎓 Njala University Pass/Fail Predictor

This project provides a machine learning-based web app to predict whether a student at Njala University has passed or failed an academic year, based on:

- GPA (Grade Point Average)
- Credit Hours
- Year Average

The application is built using **Streamlit** and uses a **Random Forest Classifier** trained on simulated student performance data.

## 📁 Files Included

- `app.py`: The Streamlit web application script
- `njala_student_data.csv`: Sample dataset simulating academic performance
- `requirements.txt`: Dependencies required to run the app

## 🚀 How to Run the App Locally

1. Clone the repository or download all files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Target Variable**: `Result` (Pass or Fail based on GPA ≥ 2.0)

## 📬 Author

Daniel Kamara  
Department of Biostatistics and Epidemiology  
Njala University
