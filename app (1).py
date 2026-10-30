import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('/Users/jsujanchowdary/Desktop/heart_pred/random_forest_model.pkl')

# App title
st.title("❤️ Heart Disease Prediction App")

# Sidebar guide
st.sidebar.title("ℹ️ Feature Value Guide")
st.sidebar.markdown("""
- **sex**: 1 = male, 0 = female  
- **cp (chest pain type)**:  
  1 = typical angina  
  2 = atypical angina  
  3 = non-anginal pain  
  4 = asymptomatic  

- **fbs (fasting blood sugar > 120 mg/dl)**: 1 = yes, 0 = no  
- **restecg**:  
  0 = normal  
  1 = ST-T wave abnormality  
  2 = left ventricular hypertrophy  

- **exang (exercise-induced angina)**: 1 = yes, 0 = no  
- **slope**:  
  0 = upsloping  
  1 = flat  
  2 = downsloping  

- **thal**:  
  3 = normal  
  6 = fixed defect  
  7 = reversible defect  
""")

# Input form
with st.form("heart_form"):
    st.subheader("Enter Patient Details")

    age = st.text_input("Age", "50")
    sex = st.text_input("Sex (1=male, 0=female)", "1")
    cp = st.text_input("Chest Pain Type (1-4)", "3")
    trestbps = st.text_input("Resting Blood Pressure (mm Hg)", "120")
    chol = st.text_input("Serum Cholestoral (mg/dl)", "200")
    fbs = st.text_input("Fasting Blood Sugar > 120 (1=yes, 0=no)", "0")
    restecg = st.text_input("Resting ECG Results (0-2)", "1")
    thalach = st.text_input("Max Heart Rate Achieved", "150")
    exang = st.text_input("Exercise Induced Angina (1=yes, 0=no)", "0")
    oldpeak = st.text_input("ST Depression (Oldpeak)", "1.0")
    slope = st.text_input("Slope (0=up, 1=flat, 2=down)", "1")
    ca = st.text_input("Number of Major Vessels (0-3)", "0")
    thal = st.text_input("Thal (3=normal, 6=fixed, 7=reversible)", "3")

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        # Feature names used during model training
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([[
            int(age),
            int(sex),
            int(cp),
            int(trestbps),
            int(chol),
            int(fbs),
            int(restecg),
            int(thalach),
            int(exang),
            float(oldpeak),
            int(slope),
            int(ca),
            int(thal)
        ]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # Show result
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Confidence: {prob:.2%})")

    except ValueError:
        st.warning("🚨 Please ensure all inputs are valid numbers.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
