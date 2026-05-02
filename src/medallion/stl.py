"""STL seasonal-trend decomposition per region."""

import pandas as pd
from statsmodels.tsa.seasonal import STL

from config import STL_COMPONENTS_FILE, SILVER_DIR


def compute_stl_per_region(
    df: pd.DataFrame,
    period: int = 4,
) -> pd.DataFrame:
    """Compute STL decomposition on quarterly median real_sqm_price per region.

    Returns DataFrame with columns:
        region, quarter_id, real_sqm_price_median, trend, seasonal, residual.
    """
    quarterly = (
        df.groupby(["region", "quarter_id"], sort=True)
        .agg(real_sqm_price_median=("real_sqm_price", "median"))
        .reset_index()
    )

    results = []
    for region, grp in quarterly.groupby("region"):
        series = grp.set_index("quarter_id")["real_sqm_price_median"].sort_index()
        series = series.ffill()

        if len(series) < 2 * period:
            continue

        stl = STL(series, period=period, robust=True)
        res = stl.fit()

        region_df = pd.DataFrame({
            "region": region,
            "quarter_id": series.index,
            "real_sqm_price_median": series.values,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        })
        results.append(region_df)

    stl_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    stl_df.to_parquet(STL_COMPONENTS_FILE, index=False)
    print(f"STL: {len(stl_df)} rows ({stl_df['region'].nunique()} regions) → {STL_COMPONENTS_FILE}")

    return stl_df
