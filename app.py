import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
lr_model = pickle.load(open("lr.pkl", "rb"))
dt_model = pickle.load(open("dt.pkl", "rb"))
rf_model = pickle.load(open("rf.pkl", "rb"))
ada_model = pickle.load(open("ada.pkl", "rb"))
xgb_model = pickle.load(open("xgb.pkl", "rb"))

# Load scaler + columns
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Student Score Predictor", layout="wide")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict exam score using multiple ML models")

# ---------------- INPUT UI ---------------- #

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 15, 30, 18)
    study_hours = st.slider("Study Hours per Day", 0, 12, 5)
    social_media = st.slider("Social Media Hours", 0, 10, 2)
    netflix = st.slider("Netflix Hours", 0, 10, 1)
    attendance = st.slider("Attendance (%)", 0, 100, 85)

with col2:
    sleep = st.slider("Sleep Hours", 0, 12, 7)
    exercise = st.slider("Exercise Frequency", 0, 7, 3)
    diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
    parental = st.selectbox("Parental Education", ["High School", "Bachelor", "Master"])
    internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    job = st.selectbox("Part Time Job", ["Yes", "No"])
    extra = st.selectbox("Extracurricular", ["Yes", "No"])

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess_input():
    input_dict = {
        'age': age,
        'study_hours_per_day': study_hours,
        'social_media_hours': social_media,
        'netflix_hours': netflix,
        'attendance_percentage': attendance,
        'sleep_hours': sleep,
        'exercise_frequency': exercise,
        'diet_quality': diet,
        'parental_education_level': parental,
        'internet_quality': internet,
        'gender': gender,
        'part_time_job': job,
        'extracurricular_participation': extra
    }

    df = pd.DataFrame([input_dict])

    # Encoding
    df['dq_e'] = df['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2})
    df['pel_e'] = df['parental_education_level'].map({'High School': 0, 'Bachelor': 1, 'Master': 2})
    df['iq_e'] = df['internet_quality'].map({'Poor': 0, 'Average': 1, 'Good': 2})

    df = pd.get_dummies(df, columns=['gender', 'part_time_job', 'extracurricular_participation'], drop_first=True)

    df = df.drop(['diet_quality', 'parental_education_level', 'internet_quality'], axis=1)
    df = df.reindex(columns=columns, fill_value=0)

    # df = df.reindex(columns=X.columns, fill_value=0)

    df_scaled = scaler.transform(df)

    return df_scaled

# ---------------- PREDICTION ---------------- #

if st.button("🔮 Predict Score"):
    input_data = preprocess_input()

    results = {
        "Linear Regression": lr_model.predict(input_data)[0],
        "Decision Tree": dt_model.predict(input_data)[0],
        "Random Forest": rf_model.predict(input_data)[0],
        "AdaBoost": ada_model.predict(input_data)[0],
        "XGBoost": xgb_model.predict(input_data)[0],
    }

    st.subheader("📊 Predicted Scores")

    result_df = pd.DataFrame({
        "Model": results.keys(),
        "Predicted Score": results.values()
    })

    st.dataframe(result_df)
