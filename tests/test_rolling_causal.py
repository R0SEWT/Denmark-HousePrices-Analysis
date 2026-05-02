import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import create_rolling_regional_features


def test_rolling_is_causal_with_outlier_year():
    """Si inyectamos un outlier masivo en el anio Y, las filas de Y no deben reflejarlo."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["A", "B"]:
        for year in range(2000, 2010):
            for _ in range(40):
                price = rng.normal(100, 5)
                if year == 2005 and region == "A":
                    price += 100_000
                rows.append({
                    "region": region,
                    "year": year,
                    "purchase_price": price,
                })

    df = pd.DataFrame(rows)
    out = create_rolling_regional_features(df, window_years=3, min_obs=10)

    row = out.query("region == 'A' and year == 2005").iloc[0]
    assert row["rolling_regional_mean"] < 200, (
        f"Causalidad rota: rolling_regional_mean={row['rolling_regional_mean']}"
    )
    assert row["rolling_regional_p90"] < 200, (
        f"Causalidad rota: rolling_regional_p90={row['rolling_regional_p90']}"
    )


def test_rolling_nan_first_years_and_counts():
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2000, 2006):
        for _ in range(10):
            rows.append({
                "region": "A",
                "year": year,
                "purchase_price": rng.normal(100, 5),
            })

    df = pd.DataFrame(rows)
    out = create_rolling_regional_features(df, window_years=3, min_obs=15)

    first_year = out.query("year == 2000")
    assert first_year["rolling_regional_mean"].isna().all()
    assert first_year["rolling_regional_p90"].isna().all()

    year_2002 = out.query("year == 2002").iloc[0]
    assert year_2002["rolling_regional_count"] == 20
