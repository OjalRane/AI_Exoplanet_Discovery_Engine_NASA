from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Features that can be used without directly exposing the final disposition.
FEATURES = [
    "koi_period","koi_time0bk","koi_impact","koi_duration",
    "koi_ingress","koi_depth","koi_ror","koi_srho","koi_prad",
    "koi_sma","koi_incl","koi_teq","koi_insol","koi_dor",
    "koi_model_dof","koi_model_chisq","koi_count","koi_num_transits",
    "koi_steff","koi_slogg","koi_smet","koi_srad","koi_smass",
    "koi_sage","koi_kepmag"
]

def load_and_prepare():
    path = ROOT / "data" / "raw" / "nasa_kepler_koi.csv"
    if not path.exists():
        raise FileNotFoundError("Run: python src/download_data.py")

    df = pd.read_csv(path)
    df = df[df["koi_disposition"].isin(
        ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
    )].copy()

    # Positive class = candidate/confirmed; negative class = false positive.
    df["target"] = df["koi_disposition"].isin(
        ["CANDIDATE", "CONFIRMED"]
    ).astype(int)

    available = [c for c in FEATURES if c in df.columns]
    X = df[available].copy()

    # Replace infinities and let model pipelines handle remaining missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    processed = X.copy()
    processed["target"] = df["target"].values
    processed.to_csv(
        ROOT / "data" / "processed" / "exoplanet_features.csv",
        index=False
    )

    return X, df["target"], df

if __name__ == "__main__":
    X, y, _ = load_and_prepare()
    print("Prepared:", X.shape)
    print(y.value_counts())
