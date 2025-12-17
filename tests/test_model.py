import os
import pytest
import pandas as pd
import joblib

def test_data_exists():
    assert os.path.exists("data/heart.csv"), "Dataset file not found!"

def test_model_exists():
    assert os.path.exists("models/model.pkl"), "Model file not found!"

def test_model_prediction():
    model = joblib.load("models/model.pkl")
    # Sample input with correct shape (13 features)
    sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    prediction = model.predict(sample_data)
    assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)"
