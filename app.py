import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model, scaler, and column names
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')

# Function for prediction
def predict_job_growth(features):
    # Convert features to DataFrame
    input_data = pd.DataFrame([features], columns=columns)
    # Scale the features
    input_data_scaled = scaler.transform(input_data)
    # Predict
    prediction = model.predict(input_data_scaled)
    return 'Growth' if prediction[0] == 1 else 'Decline'

# Create Streamlit app
st.title('Job Growth Prediction')

# Create a form for user input
st.sidebar.header('Input Features')
inputs = {}
for column in columns:
    inputs[column] = st.sidebar.number_input(f'{column}', value=0.0)

if st.sidebar.button('Predict'):
    features = [inputs[col] for col in columns]
    prediction = predict_job_growth(features)
    st.write(f'The predicted Job Growth is: {prediction}')
