import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda():
    if not os.path.exists("data/heart.csv"):
        print("Run download_data.py first!")
        return

    df = pd.read_csv("data/heart.csv")
    os.makedirs("plots", exist_ok=True)

    # 1. Histogram of Age
    plt.figure(figsize=(8, 6))
    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")
    plt.savefig("plots/age_dist.png")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title("Feature Correlation")
    plt.savefig("plots/correlation.png")
    
    # 3. Class Balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df)
    plt.title("Target Class Balance")
    plt.savefig("plots/balance.png")
    
    print("EDA plots saved to plots/ folder.")

if __name__ == "__main__":
    perform_eda()
