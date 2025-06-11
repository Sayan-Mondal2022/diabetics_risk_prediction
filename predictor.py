import joblib
import pandas as pd
import numpy as np

class DiabetesPredictor:
    def __init__(self, model_path='logreg_diabetes_model.pkl', scaler_path='diabetes_scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'BMI_Glucose', 'Age_Glucose', 'BP_Glucose'
        ]
    
    def predict(self, input_data):
        """Predict diabetes risk from input data."""
        input_df = pd.DataFrame([input_data], columns=self.feature_names[:8])  # First 8 features
        
        # Add engineered features
        input_df['BMI_Glucose'] = input_df['BMI'] * input_df['Glucose'] / 1000
        input_df['Age_Glucose'] = input_df['Age'] * input_df['Glucose'] / 1000
        input_df['BP_Glucose'] = input_df['BloodPressure'] * input_df['Glucose'] / 1000
        
        scaled_data = self.scaler.transform(input_df)
        probability = self.model.predict_proba(scaled_data)[0][1]
        
        return {
            'prediction': 'Pre-Diabetic' if probability >= 0.5 else 'Healthy',
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability)
        }
    
    def _get_risk_level(self, probability):
        """Categorize risk level."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"