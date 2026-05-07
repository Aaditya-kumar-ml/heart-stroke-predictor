import streamlit as st
import pandas as pd
import joblib

# Load model and files
model = joblib.load("KNN_heart_project.pkl")
scaler = joblib.load("scaler_heart_project.pkl")
expected_columns = joblib.load("columns_heart.pkl")

# Title
st.title("Heart Disease Prediction 💖")
st.markdown("Provide the following details")

# Inputs
age = st.slider("Age", 14, 100, 40)

sex = st.selectbox("Sex", ["MALE", "FEMALE"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fasting_BS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["YES", "NO"])

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox("Exercise-Induced Angina", ["YES", "NO"])

oldpeak = st.slider("Old Peak (ST Depression)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict button
if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_BP,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_BS == "YES" else 0,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Convert to dataframe
    input_df = pd.DataFrame([raw_input])

    # Match training columns
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Output
    if prediction == 1:
        st.error("🚨 High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")