from pathlib import Path
import pandas as pd
from flask import Flask, render_template

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)

@app.route("/")
def index():
    metrics_path = ROOT / "results" / "model_comparison.csv"
    pred_path = ROOT / "results" / "candidate_predictions.csv"

    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    predictions = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

    stats = {
        "total": len(predictions),
        "candidates": int((predictions["prediction_label"] == "Exoplanet Candidate").sum()) if len(predictions) else 0,
        "false_positive": int((predictions["prediction_label"] == "False Positive").sum()) if len(predictions) else 0,
        "best_model": metrics[0]["model"] if metrics else "Not trained",
        "best_f1": round(float(metrics[0]["f1"]) * 100, 2) if metrics else 0
    }

    return render_template(
        "index.html",
        metrics=metrics,
        stats=stats
    )

if __name__ == "__main__":
    app.run(debug=True)
