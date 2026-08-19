from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def main():
    metrics = pd.read_csv(RESULTS / "model_comparison.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics, x="model", y="f1")
    plt.title("Exoplanet Detection — F1 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close()

    pred = pd.read_csv(RESULTS / "candidate_predictions.csv")
    counts = pred["prediction_label"].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Model Prediction Distribution")
    plt.ylabel("Number of Objects")
    plt.tight_layout()
    plt.savefig(RESULTS / "prediction_distribution.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
