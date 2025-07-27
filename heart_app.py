import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#  Load and prepare data + model
@st.cache_data
def load_model():
    df = pd.read_csv(r"C:\Users\chaha\Downloads\Projects-20240722T093004Z-001\Projects\heart_disease\Heart Disease\dataset.csv")
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'target']
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = load_model()

#  UI
st.title("💓 Heart Disease Prediction")
st.write("Enter the following medical information:")

#  Inputs
age = st.slider("Age", 20, 100, 55)
sex = st.radio("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 90, 200, value=120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 400, value=240)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 210, value=150)
exang = st.radio("Exercise-Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])

#  Predict
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope'])

scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]

#  Output
if st.button("Predict"):
    if prediction == 1:
        st.error("❌ Risk Detected: Heart Disease Likely")
    else:
        st.success("✅ No Heart Disease Detected")
