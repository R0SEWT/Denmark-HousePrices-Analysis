"""Bronze layer: schema enforcement and immutable ingestion from raw data."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import BRONZE_DIR, DATA_FILE, MEDALLION_METADATA_DIR


BRONZE_OUTPUT = BRONZE_DIR / "transactions.parquet"

STRING_COLS = ["house_type", "sales_type", "region", "area", "city", "address"]
NULLABLE_INT_COLS = ["year_build", "no_rooms"]
FLOAT_COLS = [
    "purchase_price", "sqm", "sqm_price",
    "%_change_between_offer_and_purchase",
    "nom_interest_rate%", "dk_ann_infl_rate%",
    "yield_on_mortgage_credit_bonds%",
]


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    for col in NULLABLE_INT_COLS:
        if col in df.columns:
            df[col] = pd.array(pd.to_numeric(df[col], errors="coerce"), dtype="Int64")

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.array(pd.to_numeric(df[col], errors="coerce"), dtype="float64")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")

    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(4)

    return df


def _add_temporal_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["year_sale"] = df["date"].dt.year.astype("Int16")
    q = df["date"].dt.quarter
    df["quarter_id"] = (
        df["year_sale"].astype(str) + "-Q" + q.astype(str)
    )
    return df


def _add_audit_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df["_bronze_ingest_ts"] = pd.Timestamp.now(tz=timezone.utc)
    df["_source_file"] = source_file
    return df


def run_bronze(
    raw_path: Path = DATA_FILE,
    output_path: Path = BRONZE_OUTPUT,
) -> dict:
    """Ingest raw data into Bronze layer. Returns manifest dict."""
    df = pd.read_parquet(raw_path)
    source_rows = len(df)

    df = _enforce_schema(df)
    df = _add_temporal_keys(df)
    df = _add_audit_columns(df, source_file=raw_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    manifest = {
        "layer": "bronze",
        "written_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_file": str(raw_path),
        "source_row_count": source_rows,
        "output_row_count": len(df),
        "output_path": str(output_path),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    MEDALLION_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(MEDALLION_METADATA_DIR / "bronze_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Bronze: {source_rows} rows → {output_path}")
    return manifest
