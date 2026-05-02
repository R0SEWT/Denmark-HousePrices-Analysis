"""Fetch Danish postal code centroids from DAWA and write to Silver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests
import pandas as pd
from config import DAWA_POSTNUMRE_URL, POSTAL_CENTROIDS_FILE, SILVER_DIR


def fetch_postal_centroids(url: str = DAWA_POSTNUMRE_URL) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    records = []
    for r in resp.json():
        center = r.get("visueltcenter")
        kommuner = r.get("kommuner", [])
        records.append({
            "zip_code": str(r["nr"]).zfill(4),
            "postal_name": r["navn"],
            "lon": float(center[0]) if center else None,
            "lat": float(center[1]) if center else None,
            "municipality_name": kommuner[0]["navn"] if kommuner else None,
        })

    df = pd.DataFrame(records)
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    return df


def main():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_postal_centroids()
    df.to_parquet(POSTAL_CENTROIDS_FILE, index=False)
    print(f"Wrote {len(df)} postal centroids to {POSTAL_CENTROIDS_FILE}")


if __name__ == "__main__":
    main()
