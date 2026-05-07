# ❤️ Modern Premium Streamlit Frontend for Heart Stroke Prediction
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("KNN_heart_project.pkl")
scaler = joblib.load("scaler_heart_project.pkl")
expected_columns = joblib.load("columns_heart.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #dcdcdc;
    font-size: 20px;
    margin-bottom: 40px;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #111827;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    color: white;
    font-size: 22px;
    border-radius: 15px;
    border: none;
    height: 65px;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.02);
    transition: 0.2s;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #d1d5db;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered health risk prediction using Machine Learning</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About")
st.sidebar.markdown("""
### Heart Disease Prediction App

This project uses:
- KNN Machine Learning Model
- Streamlit Frontend
- Scikit-learn
- Pandas

### Features
✅ Real-time prediction  
✅ Interactive UI  
✅ ML-based analysis  
✅ Cloud Deployment  
""")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns(2)

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("👤 Patient Details")

    age = st.slider("Age", 14, 100, 40)

    sex = st.selectbox("Gender", ["MALE", "FEMALE"])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_BP = st.number_input(
        "Resting Blood Pressure",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fasting_BS = st.selectbox(
        "Fasting Blood Sugar > 120",
        ["YES", "NO"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🩺 Medical Information")

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["YES", "NO"]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Now"):

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

    input_df = pd.DataFrame([raw_input])

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            '''
            <div class="result-box" style="background:#7f1d1d;color:white;">
            🚨 High Risk of Heart Disease
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.warning("Consult a medical professional for proper diagnosis.")

    else:
        st.markdown(
            '''
            <div class="result-box" style="background:#14532d;color:white;">
            ✅ Low Risk of Heart Disease
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.success("Health indicators appear stable.")

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Developed by Aaditya kumar • Streamlit + Machine Learning ❤️</div>',
    unsafe_allow_html=True
)