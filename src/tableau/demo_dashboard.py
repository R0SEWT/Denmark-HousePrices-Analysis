"""Interactive demo dashboard — Bronze-direct analytics, 1.5M transactions.

Run:  python src/tableau/demo_dashboard.py
Then open http://127.0.0.1:8050
"""

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from config import (
    BASE_INDEX_YEAR,
    BOND_ELASTICITY_LAGS,
    BRONZE_DIR,
    CPI_BASE_YEAR,
    DATA_DIR,
    MACRO_SHOCK_YEARS,
    RESULTS_DIR,
    ROLLING_VOLATILITY_4Q,
    ROLLING_VOLATILITY_8Q,
)
from medallion.deflation import build_cpi_chain_index, deflate_to_base_year
from medallion.gold import _find_drawdown_episodes


# ══════════════════════════════════════════════════════════════════════
# STARTUP: load Bronze, deflate, pre-aggregate, free memory
# ══════════════════════════════════════════════════════════════════════

print("Loading Bronze (1.5M rows)...")
_raw = pd.read_parquet(BRONZE_DIR / "transactions.parquet")
_TOTAL_ROWS = len(_raw)

# CPI deflation (handle 2024-Q4 NaN by forward-filling inflation)
_cpi = build_cpi_chain_index(_raw)
_cpi["dk_ann_infl_rate_pct"] = _cpi["dk_ann_infl_rate_pct"].ffill()
_cpi["cpi_index_q"] = (
    (1 + _cpi["dk_ann_infl_rate_pct"] / 100) ** 0.25
).cumprod()
_raw = deflate_to_base_year(_raw, _cpi, base_year=CPI_BASE_YEAR)
print(f"CPI deflation applied ({len(_cpi)} quarters, base={CPI_BASE_YEAR})")

ALL_REGIONS = sorted(_raw["region"].unique())
ALL_HOUSE_TYPES = sorted(_raw["house_type"].unique())
SHOCK_QUARTERS = [f"{y}-Q1" for y in MACRO_SHOCK_YEARS]

# ── Pre-aggregation tables ───────────────────────────────────────────

# 1. Quarterly price by region
_qtr_price = (
    _raw.groupby(["quarter_id", "region"], sort=True)
    .agg(
        real_sqm_price_median=("real_sqm_price", "median"),
        n_transactions=("real_sqm_price", "count"),
    )
    .reset_index()
)
_qtr_price["year_sale"] = _qtr_price["quarter_id"].str[:4].astype(int)

# 2. Quarterly price by region + house_type
_qtr_price_type = (
    _raw.groupby(["quarter_id", "region", "house_type"], sort=True)
    .agg(
        real_sqm_price_median=("real_sqm_price", "median"),
        n_transactions=("real_sqm_price", "count"),
    )
    .reset_index()
)
_qtr_price_type["year_sale"] = _qtr_price_type["quarter_id"].str[:4].astype(int)

# 3. Macro indicators (one value per quarter)
_macro = (
    _raw.groupby("quarter_id", sort=True)
    .agg({
        "nom_interest_rate%": "median",
        "dk_ann_infl_rate%": "median",
        "yield_on_mortgage_credit_bonds%": "median",
    })
    .reset_index()
)
_macro["year_sale"] = _macro["quarter_id"].str[:4].astype(int)

# 4. Price index (BASE_INDEX_YEAR = 100)
_base_medians = (
    _raw[_raw["year_sale"] == BASE_INDEX_YEAR]
    .groupby("region")["real_sqm_price"]
    .median()
    .rename("base_median")
)
_price_index = _qtr_price.merge(_base_medians, on="region", how="left")
_price_index["price_index"] = (
    _price_index["real_sqm_price_median"] / _price_index["base_median"] * 100
)
_price_index = _price_index.drop(columns=["base_median"])

# Rolling volatility on price_index per region
_vol_parts = []
for region, grp in _price_index.groupby("region"):
    s = grp.set_index("quarter_id")["real_sqm_price_median"].sort_index()
    vol = pd.DataFrame({
        "region": region,
        "quarter_id": s.index,
        "rolling_4q_std": s.rolling(ROLLING_VOLATILITY_4Q, min_periods=ROLLING_VOLATILITY_4Q).std(),
        "rolling_8q_std": s.rolling(ROLLING_VOLATILITY_8Q, min_periods=ROLLING_VOLATILITY_8Q).std(),
    })
    _vol_parts.append(vol)
_volatility = pd.concat(_vol_parts, ignore_index=True)
_volatility["year_sale"] = _volatility["quarter_id"].str[:4].astype(int)

# 5. Drawdown episodes from price_index
_dd_parts = []
for region, grp in _price_index.groupby("region"):
    series = grp.set_index("quarter_id")["price_index"].sort_index().dropna()
    if len(series) < 4:
        continue
    episodes = _find_drawdown_episodes(series)
    for ep in episodes:
        ep["region"] = region
        trough_year = int(ep["trough_quarter_id"][:4])
        ep["macro_shock_label"] = next(
            (str(y) for y in MACRO_SHOCK_YEARS if abs(trough_year - y) <= 1), None
        )
    _dd_parts.extend(episodes)
_drawdown = pd.DataFrame(_dd_parts) if _dd_parts else pd.DataFrame()

# 6. OLS Bond elasticity
_elasticity = pd.DataFrame()
try:
    import statsmodels.api as sm

    _vol_national = (
        _qtr_price.groupby("quarter_id")["n_transactions"]
        .sum()
        .reset_index()
        .sort_values("quarter_id")
    )
    _vol_national["delta_pct_volume"] = _vol_national["n_transactions"].pct_change() * 100

    _yield_q = _macro[["quarter_id", "yield_on_mortgage_credit_bonds%"]].copy()
    _yield_q = _yield_q.rename(columns={"yield_on_mortgage_credit_bonds%": "yield_median"})
    _yield_q["delta_bps_yield"] = _yield_q["yield_median"].diff() * 100

    _merged_ols = _vol_national.merge(_yield_q, on="quarter_id", how="inner")

    _ols_rows = []
    for lag in BOND_ELASTICITY_LAGS:
        data = _merged_ols.copy()
        data["yield_lagged"] = data["delta_bps_yield"].shift(lag)
        data = data.dropna(subset=["delta_pct_volume", "yield_lagged"])
        if len(data) < 10:
            continue
        X = sm.add_constant(data["yield_lagged"])
        y = data["delta_pct_volume"]
        model = sm.OLS(y, X).fit()
        _ols_rows.append({
            "region": "National",
            "lag_quarters": lag,
            "beta_ols": round(float(model.params.iloc[1]), 4),
            "beta_ols_se": round(float(model.bse.iloc[1]), 4),
            "r2": round(float(model.rsquared), 4),
            "n_observations": len(data),
            "period_label": f"{data['quarter_id'].iloc[0]} - {data['quarter_id'].iloc[-1]}",
        })
    _elasticity = pd.DataFrame(_ols_rows)
except ImportError:
    print("WARNING: statsmodels not available, skipping OLS elasticity")

# ── ML results ───────────────────────────────────────────────────────

_model_results = {}
_feature_importance = []
_ml_results_path = RESULTS_DIR / "model_evaluation_summary.json"
if _ml_results_path.exists():
    with open(_ml_results_path) as f:
        _model_results = json.load(f)
    _feature_importance = _model_results.get("feature_importance", [])[:10]

# Latest median for KPI card
_latest_q = _qtr_price["quarter_id"].max()
_latest_median = _qtr_price[_qtr_price["quarter_id"] == _latest_q]["real_sqm_price_median"].median()

# Free Bronze memory
del _raw
gc.collect()

print(f"Startup complete: {_TOTAL_ROWS:,} rows -> {len(_qtr_price)} quarterly aggregates")


# ══════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d26",
    "text": "#e6e9ef",
    "muted": "#8b92a5",
    "accent": "#636efa",
    "accent2": "#ef553b",
    "accent3": "#00cc96",
    "accent4": "#ab63fa",
    "accent5": "#ffa15a",
    "grid": "#2a2d38",
}

REGION_COLORS = px.colors.qualitative.Set2[: len(ALL_REGIONS)]
HOUSE_TYPE_COLORS = px.colors.qualitative.Plotly[: len(ALL_HOUSE_TYPES)]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=48, r=24, t=40, b=36),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

MARKET_PHASES = {
    "Growth 90s": (1992, 2000, "rgba(0,204,150,0.06)"),
    "Boom 2000s": (2001, 2007, "rgba(99,110,250,0.06)"),
    "GFC":        (2008, 2012, "rgba(239,85,59,0.09)"),
    "Recovery":   (2013, 2019, "rgba(0,204,150,0.06)"),
    "Covid+":     (2020, 2024, "rgba(171,99,250,0.06)"),
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "8px",
    "padding": "12px 16px",
    "textAlign": "center",
    "minWidth": "140px",
}
PANEL_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "8px",
    "padding": "12px",
}


# ══════════════════════════════════════════════════════════════════════
# STATIC FIGURES (built once, no callbacks)
# ══════════════════════════════════════════════════════════════════════

def _build_ml_figure() -> go.Figure:
    if not _feature_importance:
        fig = go.Figure()
        fig.add_annotation(text="No ML results available", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=COLORS["muted"], size=14))
        fig.update_layout(**LAYOUT_DEFAULTS, height=300)
        return fig

    names = [f["variable"].replace("_", " ").title() for f in reversed(_feature_importance)]
    values = [f["percentage"] * 100 for f in reversed(_feature_importance)]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    layout = {**LAYOUT_DEFAULTS}
    layout["yaxis"] = dict(gridcolor=COLORS["grid"], showgrid=False)
    layout["margin"] = dict(l=180, r=60, t=10, b=36)
    fig.update_layout(**layout, height=300, xaxis_title="Importance (%)")
    return fig


_fig_ml = _build_ml_figure()


def _build_ml_metric_cards() -> list:
    if not _model_results:
        return []
    h = _model_results.get("holdout_results", {})
    metrics = [
        ("R2", f"{h.get('r2_log', 0):.4f}", "log-price"),
        ("RMSE", f"{h.get('rmse_dkk', 0):,.0f} DKK", "holdout"),
        ("MAE", f"{h.get('mae_dkk', 0):,.0f} DKK", "holdout"),
    ]
    cards = []
    for label, value, sub in metrics:
        cards.append(html.Div(style={
            "backgroundColor": COLORS["bg"], "borderRadius": "6px",
            "padding": "6px 12px", "textAlign": "center",
        }, children=[
            html.Div(value, style={"fontSize": "1rem", "fontWeight": "bold", "color": COLORS["accent3"]}),
            html.Div(f"{label} ({sub})", style={"fontSize": "0.7rem", "color": COLORS["muted"]}),
        ]))
    return cards


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _filter_year(df: pd.DataFrame, yr: list[int]) -> pd.DataFrame:
    return df[(df["year_sale"] >= yr[0]) & (df["year_sale"] <= yr[1])]


def _add_shock_lines(fig: go.Figure, quarters: list[str], show: bool) -> None:
    if not show or not fig.data:
        return
    x_vals = list(fig.data[0].x) if fig.data[0].x is not None else []
    for sq in quarters:
        if sq in x_vals:
            fig.add_shape(
                type="line", x0=sq, x1=sq, y0=0, y1=1,
                yref="paper", line=dict(color=COLORS["accent2"], width=1, dash="dot"),
                opacity=0.5,
            )
            fig.add_annotation(
                x=sq, y=1.02, yref="paper",
                text=sq[:4], showarrow=False,
                font=dict(size=10, color=COLORS["accent2"]),
            )


def _add_phase_bands(fig: go.Figure, yr: list[int], show: bool) -> None:
    if not show:
        return
    for name, (y0, y1, color) in MARKET_PHASES.items():
        if y1 < yr[0] or y0 > yr[1]:
            continue
        fig.add_vrect(
            x0=f"{max(y0, yr[0])}-Q1", x1=f"{min(y1, yr[1])}-Q4",
            fillcolor=color, line_width=0,
            annotation_text=name, annotation_font_size=9,
            annotation_font_color=COLORS["muted"],
            annotation_position="top left",
        )


# ══════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════

app = Dash(__name__, title="DK Housing — TB3 Dashboard", suppress_callback_exceptions=True)

_yr_marks = {y: str(y) for y in range(1992, 2025, 4)}

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Inter, system-ui, sans-serif",
        "color": COLORS["text"],
        "maxWidth": "1400px",
        "margin": "0 auto",
    },
    children=[
        # ── Header ───────────────────────────────────────────────
        html.Div(style={"marginBottom": "16px"}, children=[
            html.H1("Denmark Housing Prices — Hypothesis Dashboard",
                     style={"margin": "0 0 4px 0", "fontSize": "1.6rem"}),
            html.P(f"{_TOTAL_ROWS:,} transactions  ·  1992-2024  ·  Bronze-direct analytics",
                   style={"margin": 0, "color": COLORS["muted"], "fontSize": "0.85rem"}),
        ]),

        # ── KPI Cards ────────────────────────────────────────────
        html.Div(
            id="kpi-cards",
            style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div(f"{_TOTAL_ROWS:,}", style={"fontSize": "1.3rem", "fontWeight": "bold"}),
                    html.Div("Transactions", style={"fontSize": "0.75rem", "color": COLORS["muted"]}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div(f"{_latest_median:,.0f}", style={"fontSize": "1.3rem", "fontWeight": "bold"}),
                    html.Div(f"DKK/m² ({_latest_q})", style={"fontSize": "0.75rem", "color": COLORS["muted"]}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div(str(len(ALL_REGIONS)), style={"fontSize": "1.3rem", "fontWeight": "bold"}),
                    html.Div("Regions", style={"fontSize": "0.75rem", "color": COLORS["muted"]}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div(str(len(ALL_HOUSE_TYPES)), style={"fontSize": "1.3rem", "fontWeight": "bold"}),
                    html.Div("Property Types", style={"fontSize": "0.75rem", "color": COLORS["muted"]}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div(f"{len(_drawdown)}", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": COLORS["accent2"]}),
                    html.Div("Drawdown Episodes", style={"fontSize": "0.75rem", "color": COLORS["muted"]}),
                ]),
            ],
        ),

        # ── Filters ──────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "16px",
                    "flexWrap": "wrap", "alignItems": "flex-end"},
            children=[
                html.Div([
                    html.Label("Region", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                    dcc.Dropdown(
                        id="region-filter",
                        options=[{"label": r, "value": r} for r in ALL_REGIONS],
                        value=ALL_REGIONS, multi=True,
                        style={"width": "300px", "backgroundColor": COLORS["card"]},
                    ),
                ]),
                html.Div([
                    html.Label("House Type", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                    dcc.Dropdown(
                        id="house-type-filter",
                        options=[{"label": h, "value": h} for h in ALL_HOUSE_TYPES],
                        value=ALL_HOUSE_TYPES, multi=True,
                        style={"width": "300px", "backgroundColor": COLORS["card"]},
                    ),
                ]),
                html.Div([
                    html.Label("Year Range", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                    dcc.RangeSlider(
                        id="year-range", min=1992, max=2024,
                        value=[1992, 2024], marks=_yr_marks,
                        tooltip={"placement": "bottom"},
                    ),
                ], style={"width": "300px"}),
                html.Div(
                    dcc.Checklist(
                        id="overlays",
                        options=[
                            {"label": " Macro overlay", "value": "macro"},
                            {"label": " Shock markers", "value": "shocks"},
                            {"label": " Market phases", "value": "phases"},
                        ],
                        value=["shocks"],
                        inline=True,
                        style={"fontSize": "0.85rem"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px"},
                    ),
                    style={"paddingBottom": "4px"},
                ),
            ],
        ),

        # ── 2x2 Hypothesis Grid ─────────────────────────────────
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("H1 · Elasticidad Credito",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    dcc.Graph(id="chart-h1", config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("H2 · Divergencia Regional",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    dcc.Graph(id="chart-h2", config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("H3 · Resiliencia Tipologia",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    dcc.Graph(id="chart-h3", config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("Explorer · Volumen por Tipo",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    dcc.Graph(id="chart-explorer", config={"displayModeBar": False}),
                ]),
            ],
        ),

        # ── Bottom row: ML + Macro ───────────────────────────────
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px", "marginTop": "12px"},
            children=[
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("ML Model — Feature Importance (XGBoost)",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "marginBottom": "8px", "flexWrap": "wrap"},
                        children=_build_ml_metric_cards(),
                    ),
                    dcc.Graph(figure=_fig_ml, config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    html.H3("Macro Environment",
                             style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
                    dcc.Graph(id="chart-macro", config={"displayModeBar": False}),
                ]),
            ],
        ),

        # ── Elasticity Table ─────────────────────────────────────
        html.Div(
            id="elasticity-table-container",
            style={"marginTop": "12px", **PANEL_STYLE},
        ),
    ],
)


# ══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════

_SHARED_INPUTS = [
    Input("region-filter", "value"),
    Input("house-type-filter", "value"),
    Input("year-range", "value"),
    Input("overlays", "value"),
]


# ── H1: Transaction Volume + Bond Yield overlay ─────────────────────

@app.callback(Output("chart-h1", "figure"), _SHARED_INPUTS)
def update_h1(regions, house_types, yr, overlays):
    regions = regions or ALL_REGIONS
    overlays = overlays or []

    df = _qtr_price_type[
        _qtr_price_type["region"].isin(regions)
        & _qtr_price_type["house_type"].isin(house_types or ALL_HOUSE_TYPES)
    ]
    df = _filter_year(df, yr)
    agg = df.groupby(["quarter_id", "region"], as_index=False)["n_transactions"].sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, region in enumerate(sorted(agg["region"].unique())):
        rdf = agg[agg["region"] == region]
        fig.add_trace(go.Scatter(
            x=rdf["quarter_id"], y=rdf["n_transactions"],
            name=region, line=dict(color=REGION_COLORS[i % len(REGION_COLORS)], width=2),
            mode="lines",
        ), secondary_y=False)

    if "macro" in overlays:
        mdf = _filter_year(_macro, yr)
        fig.add_trace(go.Scatter(
            x=mdf["quarter_id"], y=mdf["yield_on_mortgage_credit_bonds%"],
            name="Bond Yield %", line=dict(color=COLORS["accent5"], width=1.5, dash="dash"),
            mode="lines", opacity=0.8,
        ), secondary_y=True)
        fig.update_yaxes(title_text="Bond Yield %", secondary_y=True, showgrid=False)

    fig.update_layout(**LAYOUT_DEFAULTS, height=320, showlegend=True)
    fig.update_yaxes(title_text="Transactions", secondary_y=False, gridcolor=COLORS["grid"])

    if not _elasticity.empty:
        best = _elasticity.loc[_elasticity["r2"].idxmax()]
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"OLS: B={best['beta_ols']:.3f}  R2={best['r2']:.3f}  lag={best['lag_quarters']}Q",
            showarrow=False, font=dict(size=10, color=COLORS["accent3"]),
            bgcolor="rgba(0,0,0,0.6)", borderpad=4,
        )

    _add_shock_lines(fig, SHOCK_QUARTERS, "shocks" in overlays)
    _add_phase_bands(fig, yr, "phases" in overlays)
    return fig


# ── H2: Regional Price Divergence ───────────────────────────────────

@app.callback(Output("chart-h2", "figure"), _SHARED_INPUTS)
def update_h2(regions, house_types, yr, overlays):
    regions = regions or ALL_REGIONS
    overlays = overlays or []
    df = _filter_year(_qtr_price[_qtr_price["region"].isin(regions)], yr)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, region in enumerate(sorted(df["region"].unique())):
        rdf = df[df["region"] == region]
        fig.add_trace(go.Scatter(
            x=rdf["quarter_id"], y=rdf["real_sqm_price_median"],
            name=region, line=dict(color=REGION_COLORS[i % len(REGION_COLORS)], width=2),
            mode="lines",
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=rdf["quarter_id"], y=rdf["n_transactions"],
            name=f"{region} vol.", opacity=0.15,
            marker_color=REGION_COLORS[i % len(REGION_COLORS)],
            showlegend=False,
        ), secondary_y=True)

    if "macro" in overlays:
        mdf = _filter_year(_macro, yr)
        fig.add_trace(go.Scatter(
            x=mdf["quarter_id"], y=mdf["nom_interest_rate%"],
            name="Interest %", line=dict(color=COLORS["muted"], width=1, dash="dot"),
            mode="lines", opacity=0.6,
        ), secondary_y=True)

    fig.update_layout(**LAYOUT_DEFAULTS, height=320, barmode="group")
    fig.update_yaxes(title_text="DKK/m2 (real 2024)", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="N trans.", secondary_y=True, showgrid=False)

    _add_shock_lines(fig, SHOCK_QUARTERS, "shocks" in overlays)
    _add_phase_bands(fig, yr, "phases" in overlays)
    return fig


# ── H3: Typology Resilience (price by house_type + volatility) ──────

@app.callback(Output("chart-h3", "figure"), _SHARED_INPUTS)
def update_h3(regions, house_types, yr, overlays):
    regions = regions or ALL_REGIONS
    house_types = house_types or ALL_HOUSE_TYPES
    overlays = overlays or []

    df = _qtr_price_type[
        _qtr_price_type["region"].isin(regions)
        & _qtr_price_type["house_type"].isin(house_types)
    ]
    df = _filter_year(df, yr)

    agg = (
        df.groupby(["quarter_id", "house_type"], as_index=False)
        .agg(real_sqm_price_median=("real_sqm_price_median", "median"),
             n_transactions=("n_transactions", "sum"))
    )

    fig = go.Figure()
    for i, ht in enumerate(sorted(agg["house_type"].unique())):
        hdf = agg[agg["house_type"] == ht].sort_values("quarter_id")
        fig.add_trace(go.Scatter(
            x=hdf["quarter_id"], y=hdf["real_sqm_price_median"],
            name=ht, line=dict(color=HOUSE_TYPE_COLORS[i % len(HOUSE_TYPE_COLORS)], width=2),
            mode="lines",
        ))

    if not _drawdown.empty:
        for _, ep in _drawdown.iterrows():
            peak_yr = int(ep["peak_quarter_id"][:4])
            trough_yr = int(ep["trough_quarter_id"][:4])
            if trough_yr < yr[0] or peak_yr > yr[1]:
                continue
            fig.add_vrect(
                x0=ep["peak_quarter_id"], x1=ep["trough_quarter_id"],
                fillcolor=COLORS["accent2"], opacity=0.08, line_width=0,
            )

    fig.update_layout(**LAYOUT_DEFAULTS, height=320)
    fig.update_yaxes(title_text="DKK/m2 (real 2024)")

    _add_shock_lines(fig, SHOCK_QUARTERS, "shocks" in overlays)
    _add_phase_bands(fig, yr, "phases" in overlays)
    return fig


# ── Explorer: Stacked Area Volume + optional macro ──────────────────

@app.callback(Output("chart-explorer", "figure"), _SHARED_INPUTS)
def update_explorer(regions, house_types, yr, overlays):
    regions = regions or ALL_REGIONS
    house_types = house_types or ALL_HOUSE_TYPES
    overlays = overlays or []

    df = _qtr_price_type[
        _qtr_price_type["region"].isin(regions)
        & _qtr_price_type["house_type"].isin(house_types)
    ]
    df = _filter_year(df, yr)
    agg = df.groupby(["quarter_id", "house_type"], as_index=False)["n_transactions"].sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, ht in enumerate(sorted(agg["house_type"].unique())):
        hdf = agg[agg["house_type"] == ht].sort_values("quarter_id")
        fig.add_trace(go.Scatter(
            x=hdf["quarter_id"], y=hdf["n_transactions"],
            name=ht, stackgroup="vol",
            line=dict(width=0.5, color=HOUSE_TYPE_COLORS[i % len(HOUSE_TYPE_COLORS)]),
            fillcolor=HOUSE_TYPE_COLORS[i % len(HOUSE_TYPE_COLORS)],
        ), secondary_y=False)

    if "macro" in overlays:
        mdf = _filter_year(_macro, yr)
        for col, label, color in [
            ("nom_interest_rate%", "Interest %", COLORS["muted"]),
            ("yield_on_mortgage_credit_bonds%", "Bond Yield %", COLORS["accent5"]),
            ("dk_ann_infl_rate%", "Inflation %", COLORS["accent4"]),
        ]:
            fig.add_trace(go.Scatter(
                x=mdf["quarter_id"], y=mdf[col],
                name=label, line=dict(color=color, width=1.5, dash="dash"),
                mode="lines", opacity=0.7,
            ), secondary_y=True)
        fig.update_yaxes(title_text="Rate %", secondary_y=True, showgrid=False)

    fig.update_layout(**LAYOUT_DEFAULTS, height=320)
    fig.update_yaxes(title_text="Transactions", secondary_y=False, gridcolor=COLORS["grid"])

    _add_shock_lines(fig, SHOCK_QUARTERS, "shocks" in overlays)
    _add_phase_bands(fig, yr, "phases" in overlays)
    return fig


# ── Macro Environment panel ─────────────────────────────────────────

@app.callback(
    Output("chart-macro", "figure"),
    [Input("year-range", "value"), Input("overlays", "value")],
)
def update_macro(yr, overlays):
    overlays = overlays or []
    mdf = _filter_year(_macro, yr)

    fig = go.Figure()
    traces = [
        ("nom_interest_rate%", "Interest Rate %", COLORS["accent"]),
        ("yield_on_mortgage_credit_bonds%", "Bond Yield %", COLORS["accent5"]),
        ("dk_ann_infl_rate%", "Inflation %", COLORS["accent4"]),
    ]
    for col, label, color in traces:
        fig.add_trace(go.Scatter(
            x=mdf["quarter_id"], y=mdf[col],
            name=label, line=dict(color=color, width=2),
            mode="lines",
        ))

    fig.update_layout(**LAYOUT_DEFAULTS, height=300, yaxis_title="Rate %")
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["grid"], opacity=0.5)

    _add_shock_lines(fig, SHOCK_QUARTERS, "shocks" in overlays)
    _add_phase_bands(fig, yr, "phases" in overlays)
    return fig


# ── Elasticity Table ────────────────────────────────────────────────

@app.callback(Output("elasticity-table-container", "children"), Input("region-filter", "value"))
def update_elasticity_table(_):
    if _elasticity.empty:
        return html.P(
            "OLS elasticity not available (statsmodels required)",
            style={"color": COLORS["muted"], "fontStyle": "italic"},
        )

    th_style = {"textAlign": "left", "padding": "6px 10px",
                "borderBottom": f"1px solid {COLORS['grid']}"}
    td_style = {"padding": "6px 10px"}

    rows = []
    for _, r in _elasticity.iterrows():
        rows.append(html.Tr([
            html.Td(r["region"], style=td_style),
            html.Td(f"lag {r['lag_quarters']}Q", style=td_style),
            html.Td(f"{r['beta_ols']:.4f}", style=td_style),
            html.Td(f"+/-{r['beta_ols_se']:.4f}", style=td_style),
            html.Td(f"{r['r2']:.4f}", style=td_style),
            html.Td(str(r["n_observations"]), style=td_style),
            html.Td(r["period_label"], style=td_style),
        ]))

    return [
        html.H3("Bond-Yield Elasticity (OLS)",
                 style={"margin": "0 0 6px 0", "fontSize": "0.95rem"}),
        html.P("Observational correlation — do not interpret as causal",
               style={"color": COLORS["accent2"], "fontSize": "0.78rem", "margin": "0 0 8px 0"}),
        html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"},
            children=[
                html.Thead(html.Tr([
                    html.Th(h, style=th_style)
                    for h in ["Region", "Lag", "B OLS", "SE", "R2", "N obs", "Period"]
                ])),
                html.Tbody(rows),
            ],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Dashboard -> http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
