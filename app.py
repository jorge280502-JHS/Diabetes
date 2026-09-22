import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Diabetes", layout="centered"
)

# Cargar el modelo y el escalador
model = joblib.load("xgboost.pkl")
scaler = joblib.load("scaler.pkl")

# Navegación en la barra lateral
page = st.sidebar.radio("Navegación", ["🏠 Inicio", "🧪 Predicción"])

if page == "🏠 Inicio":
  st.title("🩺 Curso : Despliegue WEB")
  st.title("🩺 Predicción de Diabetes")
  st.write(
      """
    ¡Hola ^_^! Ingresa la información del paciente para verificar si tiene diabetes.
    
    **Instrucciones:**
    - Ve a la página de Predicción.
    - Llena los datos diagnósticos del paciente.
    - Haz clic en el botón 🩺 Predecir para ver los resultados.
    """
  )

elif page == "🧪 Predicción":
  st.title("🩺 Predicción de Diabetes")
  st.write(
      "Ingresa los datos diagnósticos del paciente para obtener una predicción"
      " de diabetes."
  )

  # Formulario de entrada
  with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
      pregnancies = st.number_input("**Embarazos**", 0, 20, value=0)
      glucose = st.number_input("**Glucosa**", 50, 300, value=120)
      bp = st.number_input("**Presión Arterial**", 50, 200, value=70)
      skin = st.number_input("**Grosor de Piel**", 5, 100, value=20)

    with col2:
      insulin = st.number_input("**Insulina**", 15, 900, value=25)
      bmi = st.number_input(
          "**Índice de Masa Corporal (IMC)**", 15.0, 70.0, value=25.0, format="%.1f"
      )
      dpf = st.number_input(
          "**Función del Pedigrí de Diabetes**",
          0.1,
          5.0,
          value=0.5,
          format="%.3f",
      )
      age = st.number_input("**Edad**", 10, 120, value=30)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("**🩺 Predecir**")

  if submitted:
    # Preparar entrada
    patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    # Validación
    if not (0 <= pregnancies <= 20):
      st.error("⚠️ Los embarazos deben estar entre 0 y 20.")
    elif not (50 <= glucose <= 300):
      st.error("⚠️ La glucosa debe estar entre 50 y 300.")
    elif not (50 <= bp <= 200):
      st.error("⚠️ La presión arterial debe estar entre 50 y 200.")
    elif not (5 <= skin <= 100):
      st.error("⚠️ El grosor de piel debe estar entre 5 y 100.")
    elif not (15 <= insulin <= 900):
      st.error("⚠️ La insulina debe estar entre 15 y 900.")
    elif not (15 <= bmi <= 70):
      st.error("⚠️ El IMC debe estar entre 15 y 70.")
    elif not (0.1 <= dpf <= 5.0):
      st.error(
          "⚠️ La función del pedigrí de diabetes debe estar entre 0.1 y 5.0."
      )
    elif not (10 <= age <= 120):
      st.error("⚠️ La edad debe estar entre 10 y 120.")
    else:
      # Predicción
      Xnew = pd.DataFrame([patient])
      Xnew_scaled = scaler.transform(Xnew)

      pred = model.predict(Xnew_scaled)[0]
      probabilities = model.predict_proba(Xnew_scaled)[0]

      prob_no_diabetic = probabilities[0]
      prob_diabetic = probabilities[1]

      # Mostrar resultado
      st.markdown("---")
      st.subheader("📝 Resultado de la Predicción")

      if pred == 1:
        st.error("🔴 Diabético")
      else:
        st.success("🟢 No Diabético")

      st.write(
          f"🟢 Probabilidad de No Diabetes: **{prob_no_diabetic:.2%}**"
      )
      st.write(f"🔴 Probabilidad de Diabetes: **{prob_diabetic:.2%}**")
