# app/model.py

import pickle

def load_model():
    # Load the model from disk
    with open('RandomForest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
