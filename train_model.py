import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle

# Load the data
data = pd.read_csv('data.csv')

# Define the mappings
age_mapping = {'25-30': 0, '30-35': 0.25, '35-40': 0.5, '40-45': 0.75, '45-50': 1}
binary_mapping = {'Yes': 1, 'No': 0}
frequency_mapping = {'Not at all': 0, 'Sometimes': 0.5, 'Often': 1, 'Two or more days a week': 0.5, 'Maybe': 0.5, 'Not interested to say': 0.5}

# Map the data
data['Age'] = data['Age'].map(age_mapping)
data['Feeling sad or Tearful'] = data['Feeling sad or Tearful'].map(binary_mapping)
data['Irritable towards baby & partner'] = data['Irritable towards baby & partner'].map(frequency_mapping)
data['Trouble sleeping at night'] = data['Trouble sleeping at night'].map(frequency_mapping)
data['Problems concentrating or making decision'] = data['Problems concentrating or making decision'].map(frequency_mapping)
data['Overeating or loss of appetite'] = data['Overeating or loss of appetite'].map(frequency_mapping)
data['Feeling anxious'] = data['Feeling anxious'].map(binary_mapping)
data['Feeling of guilt'] = data['Feeling of guilt'].map(frequency_mapping)
data['Problems of bonding with baby'] = data['Problems of bonding with baby'].map(frequency_mapping)
data['Suicide attempt'] = data['Suicide attempt'].map(frequency_mapping)

# Drop the Timestamp column if it exists
if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'])

# Define features and target
X = data.drop(columns=['Feeling sad or Tearful'])

# Handle missing values in y (target variable)
y = data['Feeling sad or Tearful']
y = y.fillna(y.mode().iloc[0])  # Fill missing values with the mode

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to a pickle file
with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model has been saved to RandomForest.pkl")
