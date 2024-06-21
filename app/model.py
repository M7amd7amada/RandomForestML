import pickle

def load_model():
    with open('RandomForest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
