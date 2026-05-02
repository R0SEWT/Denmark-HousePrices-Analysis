import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medallion.bronze import _enforce_schema, _add_temporal_keys


def _make_raw_df(n=50):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="W")
    return pd.DataFrame({
        "date": (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"),
        "house_type": rng.choice(["villa", "LEJLIGHED", " rækkehus "], n),
        "sales_type": rng.choice(["Alm. Salg", "-", "familiehandel"], n),
        "region": rng.choice(["Hovedstaden", "syddanmark"], n),
        "area": ["area_" + str(i) for i in range(n)],
        "city": ["city_" + str(i) for i in range(n)],
        "address": ["addr_" + str(i) for i in range(n)],
        "purchase_price": rng.normal(2_000_000, 500_000, n),
        "sqm": rng.normal(120, 30, n),
        "sqm_price": rng.normal(15_000, 3_000, n),
        "%_change_between_offer_and_purchase": rng.normal(0, 5, n),
        "nom_interest_rate%": rng.normal(3, 1, n),
        "dk_ann_infl_rate%": rng.normal(2, 0.5, n),
        "yield_on_mortgage_credit_bonds%": rng.normal(4, 1, n),
        "year_build": rng.choice([1990, 2005, None], n),
        "no_rooms": rng.choice([3, 4, 5, None], n),
        "zip_code": rng.choice(["1000", "2100", "8000", "100"], n),
    })


def test_enforce_schema_string_cols_are_title_cased():
    df = _make_raw_df()
    result = _enforce_schema(df)
    assert (result["house_type"].str[0] == result["house_type"].str[0].str.upper()).all()
    assert not result["house_type"].str.contains(r"^\s").any()


def test_enforce_schema_float_cols_are_float64():
    df = _make_raw_df()
    result = _enforce_schema(df)
    assert result["purchase_price"].dtype == "float64"
    assert result["sqm_price"].dtype == "float64"


def test_enforce_schema_nullable_int_cols():
    df = _make_raw_df()
    result = _enforce_schema(df)
    assert str(result["year_build"].dtype) == "Int64"
    assert str(result["no_rooms"].dtype) == "Int64"


def test_enforce_schema_date_is_datetime():
    df = _make_raw_df()
    result = _enforce_schema(df)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_enforce_schema_zip_code_zero_padded():
    df = _make_raw_df()
    result = _enforce_schema(df)
    assert (result["zip_code"].str.len() == 4).all()
    assert result["zip_code"].str.match(r"^\d{4}$").all()


def test_add_temporal_keys():
    df = _make_raw_df()
    df = _enforce_schema(df)
    result = _add_temporal_keys(df)

    assert "year_sale" in result.columns
    assert "quarter_id" in result.columns
    assert result["quarter_id"].str.match(r"^\d{4}-Q[1-4]$").all()
    assert (result["year_sale"] == 2000).all()


def test_bronze_preserves_all_rows():
    df = _make_raw_df(100)
    result = _enforce_schema(df)
    assert len(result) == 100
