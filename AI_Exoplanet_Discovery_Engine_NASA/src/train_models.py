from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from preprocessing import load_and_prepare
from feature_engineering import engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    X, y, _ = load_and_prepare()
    X = engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale = ("scaler", StandardScaler())
    impute = ("imputer", SimpleImputer(strategy="median"))

    models = {
        "random_forest": Pipeline([
            impute,
            ("model", RandomForestClassifier(
                n_estimators=350, max_depth=None,
                min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=42
            ))
        ]),
        "xgboost": Pipeline([
            impute,
            ("model", XGBClassifier(
                n_estimators=350, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=42
            ))
        ]),
        "svm": Pipeline([
            impute, scale,
            ("model", SVC(C=2.0, kernel="rbf", probability=True, random_state=42))
        ]),
        "logistic_regression": Pipeline([
            impute, scale,
            ("model", LogisticRegression(
                max_iter=2500, class_weight="balanced"
            ))
        ])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")
    print("Training complete. Models saved to models/")

if __name__ == "__main__":
    main()
