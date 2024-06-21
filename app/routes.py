# app/routes.py

from flask import request, jsonify
import pandas as pd
from app.model import load_model
from app import app  # Import the 'app' object from __init__.py

# Load your machine learning model
model = load_model()

# Mapping dictionaries (if needed, keep them based on your preprocessing)
age_mapping = {'25-30': 0, '30-35': 0.25, '35-40': 0.5, '40-45': 0.75, '45-50': 1}
binary_mapping = {'Yes': 1, 'No': 0}
frequency_mapping = {'Not at all': 0, 'Sometimes': 0.5, 'Often': 1, 'Two or more days a week': 0.5, 'Maybe': 0.5, 'Not interested to say': 0.5}

# Function to preprocess JSON data
def preprocess_data(data):
    # Implement your preprocessing logic here
    return data

# Define your predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Assuming JSON input

    # Preprocess JSON data
    processed_data = preprocess_data(pd.DataFrame(data, index=[0]))

    # Extract features in the correct order
    input_data = processed_data.drop(columns=['Feeling sad or Tearful'])

    # Make prediction
    prediction = model.predict(input_data)

    # Assuming prediction is a single value or array, convert to list
    return jsonify({'prediction': prediction.tolist()})
