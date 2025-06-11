import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def load_and_preprocess_data(filepath='diabetes.csv'):
    """Load and preprocess the diabetes dataset."""
    df = pd.read_csv(filepath)
    
    # Handle zeros
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_features] = df[zero_features].replace(0, np.nan)
    
    # Fill missing values
    for col in zero_features:
        df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering
    df['BMI_Glucose'] = df['BMI'] * df['Glucose'] / 1000
    df['Age_Glucose'] = df['Age'] * df['Glucose'] / 1000
    df['BP_Glucose'] = df['BloodPressure'] * df['Glucose'] / 1000
    
    return df

def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000  # Ensure convergence
    )
    model.fit(X_train, y_train)
    return model

def save_model():
    """Main function to train and save the model."""
    df = load_and_preprocess_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = train_model(X_train_scaled, y_train)
    
    joblib.dump(model, 'logreg_diabetes_model.pkl')
    joblib.dump(scaler, 'diabetes_scaler.pkl')
    print("Model and scaler saved to disk.")

if __name__ == "__main__":
    save_model()