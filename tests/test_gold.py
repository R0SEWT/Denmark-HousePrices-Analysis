import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medallion.gold import (
    _find_drawdown_episodes,
    _label_macro_shock,
    compute_drawdown,
    compute_sqm_price_index,
    compute_transaction_volume,
    compute_volatility,
)


def _make_silver_df(n_quarters=40, n_per_quarter=50):
    rng = np.random.default_rng(42)
    rows = []
    quarters = pd.date_range("2000-01-01", periods=n_quarters, freq="QS")
    for q in quarters:
        qid = f"{q.year}-Q{(q.month - 1) // 3 + 1}"
        for _ in range(n_per_quarter):
            rows.append({
                "region": rng.choice(["Hovedstaden", "Syddanmark"]),
                "quarter_id": qid,
                "year_sale": q.year,
                "house_type": rng.choice(["Villa", "Lejlighed"]),
                "real_sqm_price": rng.normal(15_000, 3_000),
                "is_family_sale": rng.random() < 0.02,
                "is_unclassified_sale": rng.random() < 0.03,
                "yield_on_mortgage_credit_bonds%": rng.normal(4, 1),
            })
    return pd.DataFrame(rows)


def test_compute_sqm_price_index_base_year(monkeypatch):
    monkeypatch.setattr("medallion.gold.BASE_INDEX_YEAR", 2000)
    df = _make_silver_df()
    result = compute_sqm_price_index(df)
    base_rows = result[result["year_sale"] == 2000]
    base_mean = base_rows["price_index"].mean()
    assert abs(base_mean - 100.0) < 15, f"Base year mean index should be near 100, got {base_mean}"
    assert (base_rows["price_index"] > 0).all()


def test_compute_sqm_price_index_has_required_columns():
    df = _make_silver_df()
    result = compute_sqm_price_index(df)
    required = {"region", "quarter_id", "real_sqm_price_median", "price_index", "n_transactions"}
    assert required.issubset(set(result.columns))


def test_compute_transaction_volume():
    df = _make_silver_df()
    result = compute_transaction_volume(df)
    assert "n_transactions" in result.columns
    assert "pct_of_annual_total" in result.columns
    assert result["n_transactions"].sum() == len(df)


def test_compute_volatility_has_rolling_columns():
    df = _make_silver_df()
    price_idx = compute_sqm_price_index(df)
    result = compute_volatility(price_idx)
    assert "rolling_4q_std" in result.columns
    assert "rolling_8q_std" in result.columns


def test_find_drawdown_episodes_detects_crash():
    idx = pd.Index([f"2005-Q{q}" for q in range(1, 5)]
                   + [f"2006-Q{q}" for q in range(1, 5)]
                   + [f"2007-Q{q}" for q in range(1, 5)]
                   + [f"2008-Q{q}" for q in range(1, 5)]
                   + [f"2009-Q{q}" for q in range(1, 5)])

    values = list(np.linspace(100, 150, 12)) + list(np.linspace(140, 100, 8))
    series = pd.Series(values, index=idx)
    episodes = _find_drawdown_episodes(series)
    assert len(episodes) >= 1
    assert any(ep["drawdown_pct"] < -10 for ep in episodes)


def test_find_drawdown_episodes_no_crash():
    idx = pd.Index([f"2000-Q{q}" for q in range(1, 5)] * 5)
    series = pd.Series(range(20), index=idx)
    episodes = _find_drawdown_episodes(series)
    assert len(episodes) == 0


def test_label_macro_shock():
    assert _label_macro_shock("2008-Q3") == "2008"
    assert _label_macro_shock("2007-Q4") == "2008"
    assert _label_macro_shock("2020-Q2") == "2020"
    assert _label_macro_shock("2015-Q1") is None


def test_compute_drawdown_returns_dataframe():
    df = _make_silver_df()
    price_idx = compute_sqm_price_index(df)
    result = compute_drawdown(price_idx)
    assert isinstance(result, pd.DataFrame)
