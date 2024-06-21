import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('data.csv')

age_mapping = {'25-30': 0, '30-35': 0.25, '35-40': 0.5, '40-45': 0.75, '45-50': 1}
binary_mapping = {'Yes': 1, 'No': 0}
frequency_mapping = {'Not at all': 0, 'Sometimes': 0.5, 'Often': 1, 'Two or more days a week': 0.5, 'Maybe': 0.5, 'Not interested to say': 0.5}

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

if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'])

X = data[['Age', 'Irritable towards baby & partner', 'Trouble sleeping at night', 'Problems concentrating or making decision',
            'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt', 'Problems of bonding with baby', 'Suicide attempt']]

y = data['Feeling sad or Tearful']
y = y.fillna(y.mode().iloc[0])  

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model has been saved to RandomForest.pkl")
