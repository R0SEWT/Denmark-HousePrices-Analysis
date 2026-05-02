import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medallion.silver import _add_quality_flags
from medallion.deflation import build_cpi_chain_index, deflate_to_base_year


def _make_bronze_df(n=200):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="W")
    df = pd.DataFrame({
        "date": dates,
        "year_sale": dates.year,
        "quarter_id": [
            f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates
        ],
        "house_type": rng.choice(["Villa", "Lejlighed", "Rækkehus"], n),
        "sales_type": rng.choice(["Alm. Salg", "-", "Familiehandel", "nan"], n),
        "region": rng.choice(["Hovedstaden", "Syddanmark"], n),
        "zip_code": rng.choice(["1000", "2100", "8000", "9999"], n),
        "purchase_price": rng.normal(2_000_000, 500_000, n).clip(100_000),
        "sqm": np.where(rng.random(n) > 0.05, rng.normal(120, 30, n), np.nan),
        "sqm_price": rng.normal(15_000, 3_000, n),
        "year_build": np.where(rng.random(n) > 0.1, rng.integers(1900, 2020, n), pd.NA),
        "dk_ann_infl_rate%": rng.normal(2, 0.5, n),
    })
    df["year_build"] = pd.array(df["year_build"], dtype="Int64")
    return df


def test_quality_flags_are_bool_columns():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    flag_cols = [c for c in result.columns if c.startswith("is_")]
    assert len(flag_cols) >= 5
    for col in flag_cols:
        assert result[col].dtype == bool, f"{col} is not bool"


def test_quality_flags_preserve_row_count():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    assert len(result) == len(df)


def test_is_family_sale_flag():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    family_mask = result["sales_type"].str.lower().isin(["-", "familiehandel"])
    assert (result["is_family_sale"] == family_mask).all()


def test_is_pre_1995_record():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    assert not result["is_pre_1995_record"].any()

    df_old = df.copy()
    df_old["year_sale"] = 1993
    result_old = _add_quality_flags(df_old)
    assert result_old["is_pre_1995_record"].all()


def test_is_missing_sqm():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    expected = df["sqm"].isna() | (df["sqm"] <= 0)
    assert (result["is_missing_sqm"] == expected).all()


def test_is_unclassified_sale():
    df = _make_bronze_df()
    result = _add_quality_flags(df)
    unclassified = (
        (df["sales_type"] == "-")
        | (df["sales_type"].str.lower() == "nan")
        | df["sales_type"].isna()
    )
    assert (result["is_unclassified_sale"] == unclassified).all()


def test_cpi_chain_index_monotonic():
    df = _make_bronze_df()
    df["dk_ann_infl_rate%"] = 2.0
    cpi = build_cpi_chain_index(df)
    assert cpi["cpi_index_q"].is_monotonic_increasing


def test_deflation_adds_real_prices():
    df = _make_bronze_df()
    cpi = build_cpi_chain_index(df)
    result = deflate_to_base_year(df, cpi, base_year=2003)
    assert "real_purchase_price" in result.columns
    assert "real_sqm_price" in result.columns
    assert result["real_purchase_price"].dtype == "float64"
