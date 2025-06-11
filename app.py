import streamlit as st
from predictor import DiabetesPredictor

# Initialize predictor
predictor = DiabetesPredictor()

# UI Layout
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("Diabetes Risk Prediction")

# Input Form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, step=1)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=150, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
    
    submitted = st.form_submit_button("Predict")

# Handle Prediction
if submitted:
    input_data = [
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, pedigree, age
    ]
    
    result = predictor.predict(input_data)
    
    # Display Results
    st.subheader("Results")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Probability:** {result['probability']:.1%}")
    
    # Color-coded risk level
    risk_color = "red" if result['risk_level'] == "High" else "orange" if result['risk_level'] == "Medium" else "green"
    st.write(f"**Risk Level:** <span style='color:{risk_color}'>{result['risk_level']}</span>", unsafe_allow_html=True)
    
    # Progress bar
    st.progress(result['probability'])
    
    # Interpretation
    st.info("""
    **Interpretation Guide**:
    - **Healthy**: Probability < 50%  
    - **Pre-Diabetic**: Probability ≥ 50%  
    - **Risk Levels**: Low (<30%), Medium (30-70%), High (≥70%)
    """)