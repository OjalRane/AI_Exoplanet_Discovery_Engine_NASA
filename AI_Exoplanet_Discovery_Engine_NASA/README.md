# AI Exoplanet Discovery Engine — NASA Kepler KOI

This version is built around the **NASA Exoplanet Archive Kepler Objects of Interest (KOI) cumulative table**.

Source:
https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html

The project classifies KOIs as:
- 1 = CANDIDATE or CONFIRMED
- 0 = FALSE POSITIVE

`NOT DISPOSITIONED` records are excluded.

Important: `koi_score` and disposition-derived false-positive flags are intentionally excluded from the model features to reduce target leakage.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
python src/train_models.py
python src/evaluate_models.py
python src/visualization.py
python app/app.py
```

Then open http://127.0.0.1:5000

The downloader retrieves the current NASA KOI table and saves a compact project dataset locally.
