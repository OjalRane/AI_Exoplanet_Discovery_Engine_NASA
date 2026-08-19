from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "nasa_kepler_koi.csv"

COLUMNS = [
    "kepid","kepoi_name","koi_disposition",
    "koi_period","koi_time0bk","koi_impact","koi_duration",
    "koi_ingress","koi_depth","koi_ror","koi_srho",
    "koi_prad","koi_sma","koi_incl","koi_teq","koi_insol",
    "koi_dor","koi_model_dof","koi_model_chisq",
    "koi_count","koi_num_transits",
    "koi_steff","koi_slogg","koi_smet","koi_srad","koi_smass",
    "koi_sage","koi_kepmag"
]

query = (
    "select " + ",".join(COLUMNS) +
    " from cumulative where koi_disposition in "
    "('CANDIDATE','CONFIRMED','FALSE POSITIVE')"
)

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote(query) + "&format=csv"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading NASA Kepler KOI data...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    OUT.write_bytes(response.content)
    print(f"Saved: {OUT}")
    print(f"Size: {len(response.content):,} bytes")

if __name__ == "__main__":
    main()
