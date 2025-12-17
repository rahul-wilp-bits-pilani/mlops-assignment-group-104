# End-to-End MLOps Heart Disease Classification

## 1. Overview
This project implements a scalable ML pipeline to predict heart disease risk. It includes EDA, model training with MLflow tracking, containerization with Docker, and deployment manifests for Kubernetes.

## 2. Setup Instructions
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Download Data:** `python src/download_data.py`
3. **Train Model:** `python src/train.py` (Logs saved to `./mlruns`)
4. **Run EDA:** `python src/eda.py`

## 3. Experiment Tracking
We used **MLflow** to track experiments.
- **Models:** Logistic Regression, Random Forest
- **Metrics:** Accuracy, ROC-AUC
- Run `mlflow ui` to view the dashboard.

## 4. Docker & Deployment
**Build Image:**
```bash
docker build -t heart-disease-api .
