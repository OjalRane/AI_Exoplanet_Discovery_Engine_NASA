from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from preprocessing import load_and_prepare
from feature_engineering import engineer_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def main():
    X, y, raw = load_and_prepare()
    X = engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rows = []
    for path in (ROOT / "models").glob("*.pkl"):
        if path.name == "feature_columns.pkl":
            continue

        model = joblib.load(path)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, pred)
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ).to_csv(RESULTS / f"{path.stem}_confusion_matrix.csv")

        rows.append({
            "model": path.stem,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba)
        })

    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    result.to_csv(RESULTS / "model_comparison.csv", index=False)

    # Prediction summary for dashboard.
    best = result.iloc[0]["model"]
    best_model = joblib.load(ROOT / "models" / f"{best}.pkl")
    all_pred = best_model.predict(X)
    raw["prediction"] = all_pred
    raw["prediction_label"] = raw["prediction"].map({
        0: "False Positive",
        1: "Exoplanet Candidate"
    })
    raw[[
        "kepid", "kepoi_name", "koi_disposition",
        "prediction_label"
    ]].to_csv(RESULTS / "candidate_predictions.csv", index=False)

    print(result.to_string(index=False))
    print(f"Best model: {best}")

if __name__ == "__main__":
    main()
