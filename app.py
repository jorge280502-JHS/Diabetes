import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load model and scaler
model = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "🧪 Prediction"])

if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction ")
    st.write("""
    Hi there ^_^ Enter patient information to check for diabetes.
    
    **Instructions:**
    - Go to the Prediction page.
    - Fill in patient diagnostic data.
    - Click the 🩺 Predict button to see results.
    """)

elif page == "🧪 Prediction":
    st.title("🩺 Diabetes Prediction")
    st.write("Enter patient diagnostic data and get a diabetes prediction.")

    # Input form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("**Pregnancies**", 0, 20, value=0)
            glucose = st.number_input("**Glucose**", 50, 300, value=120)
            bp = st.number_input("**Blood Pressure**", 50, 200, value=70)
            skin = st.number_input("**Skin Thickness**", 5, 100, value=20)

        with col2:
            insulin = st.number_input("**Insulin**", 15, 900, value=25)
            bmi = st.number_input("**BMI**", 15.0, 70.0, value=25.0, format="%.1f")
            dpf = st.number_input("**Diabetes Pedigree Function**", 0.1, 5.0, value=0.5, format="%.3f")
            age = st.number_input("**Age**", 10, 120, value=30)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("**🩺 Predict**")

    if submitted:
        # Prepare input
        patient = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        # Validation
        if not (0 <= pregnancies <= 20):
            st.error("⚠️ Pregnancies must be between 0 and 20.")
        elif not (50 <= glucose <= 300):
            st.error("⚠️ Glucose must be between 50 and 300.")
        elif not (50 <= bp <= 200):
            st.error("⚠️ Blood Pressure must be between 50 and 200.")
        elif not (5 <= skin <= 100):
            st.error("⚠️ Skin Thickness must be between 5 and 100.")
        elif not (15 <= insulin <= 900):
            st.error("⚠️ Insulin must be between 15 and 900.")
        elif not (15 <= bmi <= 70):
            st.error("⚠️ BMI must be between 15 and 70.")
        elif not (0.1 <= dpf <= 5.0):
            st.error("⚠️ Diabetes Pedigree Function must be between 0.1 and 5.0.")
        elif not (10 <= age <= 120):
            st.error("⚠️ Age must be between 10 and 120.")
        else:
            # Prediction 
            Xnew = pd.DataFrame([patient])
            Xnew_scaled = scaler.transform(Xnew)

            # Prediction
            pred = model.predict(Xnew_scaled)[0]

            # Display result
            st.markdown("---")
            st.subheader("📝 Prediction Result")
            if pred == 1:
                st.error("🔴 Diabetic")
            else:
                st.success("🟢 Non-Diabetic")