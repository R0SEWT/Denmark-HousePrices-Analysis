"""Silver layer: enrichment, quality flags, CPI deflation, geo join, STL."""

import json
from datetime import datetime, timezone

import pandas as pd

from config import (
    BRONZE_DIR,
    FAMILY_SALE_TYPES,
    LOW_SAMPLE_THRESHOLD,
    MEDALLION_METADATA_DIR,
    SILVER_DIR,
)
from medallion.deflation import run_deflation
from medallion.geo import join_postal_centroids
from medallion.stl import compute_stl_per_region


BRONZE_INPUT = BRONZE_DIR / "transactions.parquet"
SILVER_OUTPUT = SILVER_DIR / "transactions_enriched.parquet"


def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean quality flag columns. Never filters rows."""
    df["is_pre_1995_record"] = df["year_sale"] < 1995

    sales_lower = df["sales_type"].str.lower().str.strip()
    family_lower = [s.lower() for s in FAMILY_SALE_TYPES]
    df["is_family_sale"] = sales_lower.isin(family_lower)
    df["is_unclassified_sale"] = (
        (df["sales_type"] == "-")
        | (df["sales_type"].str.lower() == "nan")
        | df["sales_type"].isna()
    )

    zip_counts = df["zip_code"].value_counts()
    low_sample_zips = set(zip_counts[zip_counts < LOW_SAMPLE_THRESHOLD].index)
    df["is_low_sample_cell"] = df["zip_code"].isin(low_sample_zips)

    df["is_missing_sqm"] = df["sqm"].isna() | (df["sqm"] <= 0)
    df["is_missing_year_build"] = df["year_build"].isna()

    percentiles = (
        df.groupby("house_type")["purchase_price"]
        .quantile([0.005, 0.995])
        .unstack()
    )
    percentiles.columns = ["p005", "p995"]

    df = df.merge(percentiles, left_on="house_type", right_index=True, how="left")
    df["is_outlier_price"] = (
        (df["purchase_price"] < df["p005"])
        | (df["purchase_price"] > df["p995"])
    )
    df = df.drop(columns=["p005", "p995"])

    flag_cols = [c for c in df.columns if c.startswith("is_")]
    for col in flag_cols:
        df[col] = df[col].astype(bool)

    return df


def run_silver(bronze_manifest: dict | None = None) -> dict:
    """Run full Silver pipeline: flags → deflation → geo → STL.

    Returns manifest dict.
    """
    df = pd.read_parquet(BRONZE_INPUT)
    source_rows = len(df)
    print(f"Silver: reading {source_rows} rows from Bronze")

    df = _add_quality_flags(df)
    print(f"  Flags added: {sum(1 for c in df.columns if c.startswith('is_'))} flag columns")

    df, cpi_table = run_deflation(df)
    print(f"  CPI deflation applied ({len(cpi_table)} quarters)")

    df = join_postal_centroids(df)

    stl_df = compute_stl_per_region(df)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_OUTPUT, index=False)
    assert len(df) == source_rows, "Silver must not lose rows"

    flag_counts = {
        col: int(df[col].sum())
        for col in df.columns if col.startswith("is_")
    }

    geo_matched = int((df["geo_join_status"] == "matched").sum())

    manifest = {
        "layer": "silver",
        "written_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_layer": "bronze",
        "source_row_count": source_rows,
        "output_row_count": len(df),
        "output_path": str(SILVER_OUTPUT),
        "columns": list(df.columns),
        "flag_counts": flag_counts,
        "geo_matched": geo_matched,
        "geo_coverage_pct": round(geo_matched / len(df) * 100, 1),
        "stl_regions": int(stl_df["region"].nunique()) if not stl_df.empty else 0,
        "caveats": [
            "dk_ann_infl_rate is annual rate reported quarterly; CPI uses (1+r)^0.25 quarterly compounding",
            "Geo centroids from DAWA (api.dataforsyningen.dk); retired postal codes may be unmatched",
        ],
    }

    MEDALLION_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEDALLION_METADATA_DIR / "silver_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Silver: {len(df)} rows → {SILVER_OUTPUT}")
    return manifest
