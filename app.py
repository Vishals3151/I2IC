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

    # Drop region column
    if 'region' in df.columns:
        df.drop(columns=['region'], inplace=True)

    # Drop rows with missing target
    df.dropna(subset=['future_expense'], inplace=True)

    # Fill missing numeric columns
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['gender', 'smoking', 'alcohol', 'disease'], drop_first=True)

    # Features and target
    X = df.drop(['future_expense'], axis=1)
    y = df['future_expense']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler, df

# Train the model
def train_model(X_train, y_train):
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model

# Streamlit UI
def run_streamlit_app():
    st.title("💊 Future Medical Expense Predictor")
    st.write("Enter patient details to predict medical costs.")

    # Load everything
    X_train, X_test, y_train, y_test, columns, scaler, df = load_and_process_data()
    model = train_model(X_train, y_train)

    # Inputs
    age = st.slider("Age", 0, 100, 30)
    disease = st.selectbox("Disease", sorted(df.filter(like='disease_').columns))
    severity = st.slider("Severity Level (1-10)", 1, 10, 5)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    smoking = st.selectbox("Smoker?", ['No', 'Yes'])
    alcohol = st.selectbox("Alcohol Consumer?", ['No', 'Yes'])
    insurance = st.selectbox("Has Insurance?", [0, 1])

    # Prepare input
    input_data = {
        'age': age,
        'severity': severity,
        'insurance': insurance,
        'gender_Male': 1 if gender == 'Male' else 0,
        'smoking_Yes': 1 if smoking == 'Yes' else 0,
        'alcohol_Yes': 1 if alcohol == 'Yes' else 0,
    }

    # Add all disease dummy cols as 0
    for col in df.columns:
        if col.startswith("disease_"):
            input_data[col] = 1 if col == disease else 0

    # Add missing dummy cols with 0
    for col in columns:
        if col not in input_data:
            input_data[col] = 0

    # Final DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"🧾 Predicted Medical Expense: ₹{prediction:,.2f}")

if __name__ == "__main__":
    run_streamlit_app()
