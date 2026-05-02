"""CPI chain-linking and price deflation utilities."""

import pandas as pd

from config import CPI_BASE_YEAR, CPI_INDEX_FILE, SILVER_DIR


def build_cpi_chain_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build quarterly CPI index from dk_ann_infl_rate% in the data.

    Returns DataFrame with columns: quarter_id, cpi_index_q, dk_ann_infl_rate_pct.
    The index is chained from 1.0 at the earliest quarter, using quarterly
    compounding: (1 + annual_rate/100)^(1/4) per quarter.
    """
    quarterly = (
        df.groupby("quarter_id", sort=True)["dk_ann_infl_rate%"]
        .median()
        .reset_index()
        .rename(columns={"dk_ann_infl_rate%": "dk_ann_infl_rate_pct"})
    )
    quarterly = quarterly.sort_values("quarter_id").reset_index(drop=True)

    quarterly_factor = (1 + quarterly["dk_ann_infl_rate_pct"] / 100) ** 0.25
    quarterly["cpi_index_q"] = quarterly_factor.cumprod()

    return quarterly


def deflate_to_base_year(
    df: pd.DataFrame,
    cpi_table: pd.DataFrame,
    base_year: int = CPI_BASE_YEAR,
) -> pd.DataFrame:
    """Add real_purchase_price and real_sqm_price deflated to base_year DKK."""
    base_quarters = cpi_table[cpi_table["quarter_id"].str.startswith(str(base_year))]
    if base_quarters.empty:
        last_q = str(cpi_table["quarter_id"].max())
        cpi_base = float(
            cpi_table[cpi_table["quarter_id"] == last_q]["cpi_index_q"].values[0]
        )
    else:
        cpi_base = float(base_quarters["cpi_index_q"].values[-1])

    df = df.merge(
        cpi_table[["quarter_id", "cpi_index_q"]],
        on="quarter_id",
        how="left",
    )

    deflator = cpi_base / df["cpi_index_q"]
    df["real_purchase_price"] = (df["purchase_price"] * deflator).astype("float64")
    df["real_sqm_price"] = (df["sqm_price"] * deflator).astype("float64")

    return df


def run_deflation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build CPI index, deflate prices, save CPI table. Returns (df, cpi_table)."""
    cpi_table = build_cpi_chain_index(df)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    cpi_table.to_parquet(CPI_INDEX_FILE, index=False)
    print(f"CPI index: {len(cpi_table)} quarters → {CPI_INDEX_FILE}")

    df = deflate_to_base_year(df, cpi_table)
    return df, cpi_table
