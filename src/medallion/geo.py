"""Geographic enrichment: postal code centroid join."""

import pandas as pd

from config import POSTAL_CENTROIDS_FILE


def join_postal_centroids(
    df: pd.DataFrame,
    centroids_path=POSTAL_CENTROIDS_FILE,
) -> pd.DataFrame:
    """Left-join postal centroids on zip_code. Adds lat, lon, municipality_name."""
    centroids = pd.read_parquet(centroids_path)

    df = df.merge(
        centroids[["zip_code", "lat", "lon", "municipality_name"]],
        on="zip_code",
        how="left",
    )

    df["geo_join_status"] = "matched"
    df.loc[df["lat"].isna(), "geo_join_status"] = "unmatched"

    matched = (df["geo_join_status"] == "matched").sum()
    total = len(df)
    print(f"Geo join: {matched}/{total} matched ({matched/total*100:.1f}%)")

    return df
