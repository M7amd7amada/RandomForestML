from flask import request, jsonify
import pandas as pd
from app.model import load_model
from app import app  

model = load_model()

age_mapping = {'25-30': 0, '30-35': 0.25, '35-40': 0.5, '40-45': 0.75, '45-50': 1}
binary_mapping = {'Yes': 1, 'No': 0}
frequency_mapping = {'Not at all': 0, 'Sometimes': 0.5, 'Often': 1, 'Two or more days a week': 0.5, 'Maybe': 0.5, 'Not interested to say': 0.5}

def preprocess_data(data):
    data['Age'] = data['Age'].map(age_mapping)
    data['Feeling anxious'] = data['Feeling anxious'].map(binary_mapping)
    data['Trouble sleeping at night'] = data['Trouble sleeping at night'].map(frequency_mapping)
    data['Problems concentrating or making decision'] = data['Problems concentrating or making decision'].map(frequency_mapping)
    data['Irritable towards baby & partner'] = data['Irritable towards baby & partner'].map(frequency_mapping)
    data['Overeating or loss of appetite'] = data['Overeating or loss of appetite'].map(frequency_mapping)
    data['Feeling of guilt'] = data['Feeling of guilt'].map(frequency_mapping)
    data['Problems of bonding with baby'] = data['Problems of bonding with baby'].map(frequency_mapping)
    data['Suicide attempt'] = data['Suicide attempt'].map(frequency_mapping)
    return data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  

    processed_data = preprocess_data(pd.DataFrame(data, index=[0]))

    input_data = processed_data[['Age', 'Irritable towards baby & partner', 'Trouble sleeping at night', 'Problems concentrating or making decision',
                                    'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt', 'Problems of bonding with baby', 'Suicide attempt']]

    probabilities = model.predict_proba(input_data)
    
    probability_positive_class = probabilities[0][1]

    probability_percentage = round(probability_positive_class * 100, 2)

    return jsonify({'probability_percentage': probability_percentage})

