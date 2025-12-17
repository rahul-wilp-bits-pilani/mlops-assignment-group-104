import pandas as pd
import os

def download_data():
    # URL for Heart Disease UCI dataset (using a direct clean version for simplicity)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names based on UCI documentation
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    print("Downloading dataset...")
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # Simple preprocessing: Drop NaNs and map target to 0 (no disease) and 1 (disease)
    df.dropna(inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/heart.csv", index=False)
    print("Data saved to data/heart.csv")

if __name__ == "__main__":
    download_data()
