import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import os

def train():
    df = pd.read_csv("data/heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize MLflow
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }
    
    best_model = None
    best_acc = 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            # Logging
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", auc)
            
            mlflow.sklearn.log_model(model, name)
            print(f"{name} - Accuracy: {acc}")
            
            if acc > best_acc:
                best_acc = acc
                best_model = model

    # Save best model for production
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    print("Best model saved to models/model.pkl")

if __name__ == "__main__":
    train()
