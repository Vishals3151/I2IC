import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("i2ic.csv")

    # Drop unwanted column if exists
    df.drop(columns=['region'], inplace=True, errors='ignore')

    # Drop rows with missing target
    df.dropna(subset=['future_expense'], inplace=True)

    # Fill missing numeric values with column means
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'smoking', 'alcohol', 'disease'], drop_first=True)

    # Define features and target
    X = df.drop(columns=['future_expense'])
    y = df['future_expense']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler, df

# Train the SVR model
def train_model(X_train, y_train):
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model

# Main Streamlit app
def run_streamlit_app():
    st.set_page_config(page_title="Medical Expense Predictor", layout="centered")
    st.title("💊 Future Medical Expense Predictor")
    st.write("Predict future medical expenses based on patient details.")

    # Load data and model
    X_train, X_test, y_train, y_test, columns, scaler, df = load_and_process_data()
    model = train_model(X_train, y_train)

    # Input fields
    age = st.slider("Age", 0, 100, 30)
    severity = st.slider("Severity Level (1-10)", 1, 10, 5)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    smoking = st.selectbox("Smoker?", ['No', 'Yes'])
    alcohol = st.selectbox("Alcohol Consumer?", ['No', 'Yes'])
    insurance = st.selectbox("Has Insurance?", [0, 1])

    # Choose disease from encoded column names
    disease_options = [col for col in df.columns if col.startswith("disease_")]
    disease = st.selectbox("Disease", sorted(disease_options))

    # Prepare input dictionary
    input_data = {
        'age': age,
        'severity': severity,
        'insurance': insurance,
        'gender_Male': int(gender == 'Male'),
        'smoking_Yes': int(smoking == 'Yes'),
        'alcohol_Yes': int(alcohol == 'Yes')
    }

    # Initialize all disease columns to 0, set selected one to 1
    for col in disease_options:
        input_data[col] = int(col == disease)

    # Add missing columns from training data
    for col in columns:
        if col not in input_data:
            input_data[col] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"🧾 Predicted Future Medical Expense: ₹{prediction:,.2f}")

if __name__ == "__main__":
    run_streamlit_app()
