"""Gold layer: aggregated KPI tables for Tableau dashboard."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    BASE_INDEX_YEAR,
    BOND_ELASTICITY_LAGS,
    GOLD_DIR,
    MACRO_SHOCK_YEARS,
    MEDALLION_METADATA_DIR,
    ROLLING_VOLATILITY_4Q,
    ROLLING_VOLATILITY_8Q,
    SILVER_DIR,
    STL_COMPONENTS_FILE,
)


SILVER_INPUT = SILVER_DIR / "transactions_enriched.parquet"

DEFAULT_EXCLUDE_FLAGS = ["is_family_sale", "is_unclassified_sale"]


def _load_silver(exclude_flags: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(SILVER_INPUT)
    if exclude_flags:
        for flag in exclude_flags:
            if flag in df.columns:
                df = df[~df[flag]]
    return df


# ── KPI 1: Regional Price Index (1992=100) ──────────────────────────


def compute_sqm_price_index(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = _load_silver(DEFAULT_EXCLUDE_FLAGS)

    df = df.dropna(subset=["real_sqm_price"])

    quarterly = (
        df.groupby(["region", "quarter_id"], sort=True)
        .agg(
            real_sqm_price_median=("real_sqm_price", "median"),
            n_transactions=("real_sqm_price", "count"),
        )
        .reset_index()
    )

    base_medians = (
        df[df["year_sale"] == BASE_INDEX_YEAR]
        .groupby("region")["real_sqm_price"]
        .median()
        .rename("base_median")
    )

    quarterly = quarterly.merge(base_medians, on="region", how="left")
    quarterly["price_index"] = (
        quarterly["real_sqm_price_median"] / quarterly["base_median"] * 100
    )
    quarterly = quarterly.drop(columns=["base_median"])

    quarterly["year_sale"] = quarterly["quarter_id"].str[:4].astype(int)
    quarterly["quarter_num"] = quarterly["quarter_id"].str[-1].astype(int)

    if STL_COMPONENTS_FILE.exists():
        stl = pd.read_parquet(STL_COMPONENTS_FILE)
        quarterly = quarterly.merge(
            stl[["region", "quarter_id", "trend"]].rename(
                columns={"trend": "price_index_stl_trend"}
            ),
            on=["region", "quarter_id"],
            how="left",
        )

    output = GOLD_DIR / "kpi_sqm_price_index.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    quarterly.to_parquet(output, index=False)
    print(f"KPI price_index: {len(quarterly)} rows, {quarterly['region'].nunique()} regions → {output}")
    return quarterly


# ── KPI 2: Transaction Volume ────────────────────────────────────────


def compute_transaction_volume(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = _load_silver(exclude_flags=None)

    volume = (
        df.groupby(["region", "quarter_id", "house_type"], sort=True)
        .size()
        .reset_index(name="n_transactions")
    )

    volume["year_sale"] = volume["quarter_id"].str[:4].astype(int)
    volume["quarter_num"] = volume["quarter_id"].str[-1].astype(int)

    annual_totals = (
        volume.groupby(["region", "year_sale"])["n_transactions"]
        .transform("sum")
    )
    volume["pct_of_annual_total"] = (
        volume["n_transactions"] / annual_totals * 100
    ).round(2)

    output = GOLD_DIR / "kpi_transaction_volume.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    volume.to_parquet(output, index=False)
    print(f"KPI volume: {len(volume)} rows → {output}")
    return volume


# ── KPI 3: Peak-to-Trough Drawdown ──────────────────────────────────


def _find_drawdown_episodes(series: pd.Series) -> list[dict]:
    """Greedy drawdown scan: find episodes where price drops from a peak."""
    episodes = []
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax

    in_drawdown = False
    peak_idx = None
    trough_idx = None
    trough_val = 0.0

    for i, (idx, dd) in enumerate(drawdown.items()):
        if dd < -0.02 and not in_drawdown:
            in_drawdown = True
            peak_idx = cummax[:idx].idxmax() if i > 0 else idx
            trough_idx = idx
            trough_val = dd
        elif in_drawdown and dd < trough_val:
            trough_idx = idx
            trough_val = dd
        elif in_drawdown and dd > -0.01:
            episodes.append({
                "peak_quarter_id": peak_idx,
                "trough_quarter_id": trough_idx,
                "peak_index": float(series[peak_idx]),
                "trough_index": float(series[trough_idx]),
                "drawdown_pct": round(float(trough_val) * 100, 2),
            })
            in_drawdown = False
            trough_val = 0.0

    if in_drawdown:
        episodes.append({
            "peak_quarter_id": peak_idx,
            "trough_quarter_id": trough_idx,
            "peak_index": float(series[peak_idx]),
            "trough_index": float(series[trough_idx]),
            "drawdown_pct": round(float(trough_val) * 100, 2),
        })

    return episodes


def _label_macro_shock(quarter_id: str) -> str | None:
    year = int(quarter_id[:4])
    for shock_year in MACRO_SHOCK_YEARS:
        if abs(year - shock_year) <= 1:
            return str(shock_year)
    return None


def compute_drawdown(price_index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if price_index_df is None:
        path = GOLD_DIR / "kpi_sqm_price_index.parquet"
        price_index_df = pd.read_parquet(path)

    all_episodes = []

    for (region,), grp in price_index_df.groupby(["region"]):
        series = grp.set_index("quarter_id")["price_index"].sort_index()
        episodes = _find_drawdown_episodes(series)
        for ep in episodes:
            ep["region"] = region
            ep["house_type"] = "All"
            peak_q = ep["peak_quarter_id"]
            trough_q = ep["trough_quarter_id"]
            peak_pos = list(series.index).index(peak_q) if peak_q in series.index else 0
            trough_pos = list(series.index).index(trough_q) if trough_q in series.index else 0
            ep["drawdown_duration_quarters"] = trough_pos - peak_pos
            ep["macro_shock_label"] = _label_macro_shock(trough_q)
            all_episodes.append(ep)

    result = pd.DataFrame(all_episodes) if all_episodes else pd.DataFrame()

    output = GOLD_DIR / "kpi_drawdown.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"KPI drawdown: {len(result)} episodes → {output}")
    return result


# ── KPI 4: Inter-Quarterly Volatility ────────────────────────────────


def compute_volatility(price_index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if price_index_df is None:
        path = GOLD_DIR / "kpi_sqm_price_index.parquet"
        price_index_df = pd.read_parquet(path)

    results = []
    for region, grp in price_index_df.groupby("region"):
        series = grp.set_index("quarter_id")["real_sqm_price_median"].sort_index()

        vol_df = pd.DataFrame({
            "region": region,
            "quarter_id": series.index,
            "rolling_4q_std": series.rolling(
                ROLLING_VOLATILITY_4Q, min_periods=ROLLING_VOLATILITY_4Q
            ).std(),
            "rolling_8q_std": series.rolling(
                ROLLING_VOLATILITY_8Q, min_periods=ROLLING_VOLATILITY_8Q
            ).std(),
        })
        results.append(vol_df)

    result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    result["year_sale"] = result["quarter_id"].str[:4].astype(int)
    result["quarter_num"] = result["quarter_id"].str[-1].astype(int)

    output = GOLD_DIR / "kpi_volatility.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"KPI volatility: {len(result)} rows → {output}")
    return result


# ── KPI 5: Volume-Bond Yield Elasticity ──────────────────────────────


def compute_bond_elasticity(
    volume_df: pd.DataFrame | None = None,
    silver_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if volume_df is None:
        path = GOLD_DIR / "kpi_transaction_volume.parquet"
        volume_df = pd.read_parquet(path)

    if silver_df is None:
        silver_df = pd.read_parquet(SILVER_INPUT)

    yield_quarterly = (
        silver_df.groupby("quarter_id")["yield_on_mortgage_credit_bonds%"]
        .median()
        .reset_index()
        .rename(columns={"yield_on_mortgage_credit_bonds%": "yield_median"})
        .sort_values("quarter_id")
    )
    yield_quarterly["delta_bps_yield"] = yield_quarterly["yield_median"].diff() * 100

    vol_national = (
        volume_df.groupby("quarter_id")["n_transactions"]
        .sum()
        .reset_index()
        .sort_values("quarter_id")
    )
    vol_national["delta_pct_volume"] = (
        vol_national["n_transactions"].pct_change() * 100
    )

    merged = vol_national.merge(yield_quarterly, on="quarter_id", how="inner")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("WARNING: statsmodels not available, skipping OLS elasticity")
        return pd.DataFrame()

    results = []
    for lag in BOND_ELASTICITY_LAGS:
        data = merged.copy()
        data["yield_lagged"] = data["delta_bps_yield"].shift(lag)
        data = data.dropna(subset=["delta_pct_volume", "yield_lagged"])

        if len(data) < 10:
            continue

        X = sm.add_constant(data["yield_lagged"])
        y = data["delta_pct_volume"]
        model = sm.OLS(y, X).fit()

        results.append({
            "region": "National",
            "lag_quarters": lag,
            "beta_ols": round(float(model.params.iloc[1]), 4),
            "beta_ols_se": round(float(model.bse.iloc[1]), 4),
            "r2": round(float(model.rsquared), 4),
            "n_observations": len(data),
            "period_label": f"{data['quarter_id'].iloc[0]}–{data['quarter_id'].iloc[-1]}",
            "causal_warning": "confounded_OLS",
        })

    result = pd.DataFrame(results) if results else pd.DataFrame()

    output = GOLD_DIR / "kpi_bond_elasticity.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"KPI elasticity: {len(result)} lag configs → {output}")
    return result


# ── Gold Orchestrator ─────────────────────────────────────────────────


def run_gold(silver_manifest: dict | None = None) -> dict:
    """Run all Gold KPI computations. Returns manifest dict."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    price_idx = compute_sqm_price_index()
    volume = compute_transaction_volume()
    drawdown = compute_drawdown(price_idx)
    volatility = compute_volatility(price_idx)
    elasticity = compute_bond_elasticity(volume)

    manifest = {
        "layer": "gold",
        "written_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_layer": "silver",
        "kpis": {
            "price_index": {"rows": len(price_idx), "file": "kpi_sqm_price_index.parquet"},
            "volume": {"rows": len(volume), "file": "kpi_transaction_volume.parquet"},
            "drawdown": {"episodes": len(drawdown), "file": "kpi_drawdown.parquet"},
            "volatility": {"rows": len(volatility), "file": "kpi_volatility.parquet"},
            "elasticity": {"rows": len(elasticity), "file": "kpi_bond_elasticity.parquet"},
        },
        "caveats": [
            "bond_elasticity is observational OLS — do not interpret as causal",
            "drawdown episodes use greedy scan with 2% entry / 1% exit thresholds",
        ],
    }

    MEDALLION_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEDALLION_METADATA_DIR / "gold_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGold: 5 KPIs computed → {GOLD_DIR}")
    return manifest
