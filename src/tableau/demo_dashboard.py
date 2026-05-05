"""Interactive demo dashboard — stand-in while Tableau is unavailable.

Run:  python src/tableau/demo_dashboard.py
Then open http://127.0.0.1:8050
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from config import GOLD_DIR, MACRO_SHOCK_YEARS

# ── Load Gold KPIs ───────────────────────────────────────────────────

_price = pd.read_parquet(GOLD_DIR / "kpi_sqm_price_index.parquet")
_volume = pd.read_parquet(GOLD_DIR / "kpi_transaction_volume.parquet")
_volatility = pd.read_parquet(GOLD_DIR / "kpi_volatility.parquet")

try:
    _drawdown = pd.read_parquet(GOLD_DIR / "kpi_drawdown.parquet")
except Exception:
    _drawdown = pd.DataFrame()

try:
    _elasticity = pd.read_parquet(GOLD_DIR / "kpi_bond_elasticity.parquet")
except Exception:
    _elasticity = pd.DataFrame()

ALL_REGIONS = sorted(_price["region"].unique())
ALL_HOUSE_TYPES = sorted(_volume["house_type"].unique())

SHOCK_QUARTERS = [f"{y}-Q1" for y in MACRO_SHOCK_YEARS]

# ── Colour palette ───────────────────────────────────────────────────

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d26",
    "text": "#e6e9ef",
    "muted": "#8b92a5",
    "accent": "#636efa",
    "accent2": "#ef553b",
    "accent3": "#00cc96",
    "grid": "#2a2d38",
}

REGION_COLORS = px.colors.qualitative.Set2[: len(ALL_REGIONS)]

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

# ── App ──────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    title="DK Housing — TB3 Demo",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Inter, system-ui, sans-serif",
        "color": COLORS["text"],
    },
    children=[
        # Header
        html.Div(
            style={"marginBottom": "24px"},
            children=[
                html.H1(
                    "Denmark Housing Prices — Hypothesis Dashboard",
                    style={"margin": "0 0 4px 0", "fontSize": "1.6rem"},
                ),
                html.P(
                    "Demo interactivo (TB3) — datos Gold layer",
                    style={"margin": 0, "color": COLORS["muted"], "fontSize": "0.9rem"},
                ),
            ],
        ),
        # Filters
        html.Div(
            style={
                "display": "flex",
                "gap": "24px",
                "marginBottom": "24px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    [
                        html.Label("Region", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[{"label": r, "value": r} for r in ALL_REGIONS],
                            value=ALL_REGIONS,
                            multi=True,
                            style={"width": "320px", "backgroundColor": COLORS["card"]},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("House Type", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                        dcc.Dropdown(
                            id="house-type-filter",
                            options=[{"label": h, "value": h} for h in ALL_HOUSE_TYPES],
                            value=ALL_HOUSE_TYPES,
                            multi=True,
                            style={"width": "240px", "backgroundColor": COLORS["card"]},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Shock markers", style={"fontSize": "0.8rem", "color": COLORS["muted"]}),
                        dcc.Checklist(
                            id="show-shocks",
                            options=[{"label": " Show macro shocks", "value": "yes"}],
                            value=["yes"],
                            style={"paddingTop": "6px"},
                        ),
                    ]
                ),
            ],
        ),
        # 2×2 grid
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
            },
            children=[
                # H1
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H3("H1 · Elasticidad Crédito", style={"margin": "0 0 8px 0", "fontSize": "1rem"}),
                        dcc.Graph(id="chart-h1", config={"displayModeBar": False}),
                    ],
                ),
                # H2
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H3("H2 · Divergencia Regional", style={"margin": "0 0 8px 0", "fontSize": "1rem"}),
                        dcc.Graph(id="chart-h2", config={"displayModeBar": False}),
                    ],
                ),
                # H3
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H3("H3 · Resiliencia Tipología", style={"margin": "0 0 8px 0", "fontSize": "1rem"}),
                        dcc.Graph(id="chart-h3", config={"displayModeBar": False}),
                    ],
                ),
                # Explorer
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H3("Explorer · Volumen por Tipo", style={"margin": "0 0 8px 0", "fontSize": "1rem"}),
                        dcc.Graph(id="chart-explorer", config={"displayModeBar": False}),
                    ],
                ),
            ],
        ),
        # Elasticity table
        html.Div(
            id="elasticity-table-container",
            style={
                "marginTop": "16px",
                "backgroundColor": COLORS["card"],
                "borderRadius": "8px",
                "padding": "16px",
            },
        ),
    ],
)


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


# ── H1: Transaction Volume (proxy for credit elasticity) ────────────

@app.callback(Output("chart-h1", "figure"), [Input("region-filter", "value"), Input("show-shocks", "value")])
def update_h1(regions, show_shocks):
    df = _volume[_volume["region"].isin(regions or ALL_REGIONS)]
    agg = df.groupby(["quarter_id", "region"], as_index=False)["n_transactions"].sum().sort_values("quarter_id")

    fig = px.line(
        agg, x="quarter_id", y="n_transactions", color="region",
        labels={"quarter_id": "", "n_transactions": "Transacciones", "region": ""},
        color_discrete_sequence=REGION_COLORS,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=340, showlegend=True)
    fig.update_traces(line_width=2)

    _add_shock_lines(fig, SHOCK_QUARTERS, bool(show_shocks))

    if len(_elasticity):
        best = _elasticity.loc[_elasticity["r2"].idxmax()]
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"OLS β={best['beta_ols']:.3f}  R²={best['r2']:.3f}  (lag {best['lag_quarters']}Q)",
            showarrow=False, font=dict(size=11, color=COLORS["accent3"]),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4,
        )

    return fig


# ── H2: Regional Price Divergence ───────────────────────────────────

@app.callback(Output("chart-h2", "figure"), [Input("region-filter", "value"), Input("show-shocks", "value")])
def update_h2(regions, show_shocks):
    df = _price[_price["region"].isin(regions or ALL_REGIONS)].sort_values("quarter_id")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, region in enumerate(sorted(df["region"].unique())):
        rdf = df[df["region"] == region]
        fig.add_trace(
            go.Scatter(
                x=rdf["quarter_id"], y=rdf["real_sqm_price_median"],
                name=region, line=dict(color=REGION_COLORS[i % len(REGION_COLORS)], width=2),
                mode="lines",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=rdf["quarter_id"], y=rdf["n_transactions"],
                name=f"{region} vol.", opacity=0.25,
                marker_color=REGION_COLORS[i % len(REGION_COLORS)],
                showlegend=False,
            ),
            secondary_y=True,
        )

    fig.update_layout(**LAYOUT_DEFAULTS, height=340, barmode="group")
    fig.update_yaxes(title_text="DKK/m² (real)", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="N trans.", secondary_y=True, showgrid=False)

    _add_shock_lines(fig, SHOCK_QUARTERS, bool(show_shocks))
    return fig


# ── H3: Volatility ──────────────────────────────────────────────────

@app.callback(Output("chart-h3", "figure"), [Input("region-filter", "value"), Input("show-shocks", "value")])
def update_h3(regions, show_shocks):
    df = _volatility[_volatility["region"].isin(regions or ALL_REGIONS)].sort_values("quarter_id")

    fig = go.Figure()
    for i, region in enumerate(sorted(df["region"].unique())):
        rdf = df[df["region"] == region]
        fig.add_trace(go.Scatter(
            x=rdf["quarter_id"], y=rdf["rolling_4q_std"],
            name=f"{region} 4Q", line=dict(color=REGION_COLORS[i % len(REGION_COLORS)], width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=rdf["quarter_id"], y=rdf["rolling_8q_std"],
            name=f"{region} 8Q", line=dict(color=REGION_COLORS[i % len(REGION_COLORS)], width=1, dash="dash"),
            mode="lines",
        ))

    fig.update_layout(**LAYOUT_DEFAULTS, height=340)
    fig.update_yaxes(title_text="Std Dev (DKK/m²)")

    if len(_drawdown):
        for _, ep in _drawdown.iterrows():
            fig.add_vrect(
                x0=ep["peak_quarter_id"], x1=ep["trough_quarter_id"],
                fillcolor=COLORS["accent2"], opacity=0.12, line_width=0,
                annotation_text=f"{ep['drawdown_pct']:.0f}%",
                annotation_font_size=9,
            )

    _add_shock_lines(fig, SHOCK_QUARTERS, bool(show_shocks))
    return fig


# ── Explorer: Stacked Volume by House Type ──────────────────────────

@app.callback(
    Output("chart-explorer", "figure"),
    [Input("region-filter", "value"), Input("house-type-filter", "value"), Input("show-shocks", "value")],
)
def update_explorer(regions, house_types, show_shocks):
    df = _volume[
        _volume["region"].isin(regions or ALL_REGIONS)
        & _volume["house_type"].isin(house_types or ALL_HOUSE_TYPES)
    ].sort_values("quarter_id")

    agg = df.groupby(["quarter_id", "house_type"], as_index=False)["n_transactions"].sum()

    fig = px.area(
        agg, x="quarter_id", y="n_transactions", color="house_type",
        labels={"quarter_id": "", "n_transactions": "Transacciones", "house_type": ""},
        color_discrete_sequence=[COLORS["accent"], COLORS["accent3"]],
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=340)
    _add_shock_lines(fig, SHOCK_QUARTERS, bool(show_shocks))
    return fig


# ── Elasticity summary table ────────────────────────────────────────

@app.callback(Output("elasticity-table-container", "children"), Input("region-filter", "value"))
def update_elasticity_table(_):
    if _elasticity.empty:
        return html.P(
            "⚠ kpi_bond_elasticity.parquet no disponible — ejecutar pipeline Silver → Gold para generar.",
            style={"color": COLORS["muted"], "fontStyle": "italic"},
        )

    rows = []
    for _, r in _elasticity.iterrows():
        rows.append(
            html.Tr([
                html.Td(r.get("region", "")),
                html.Td(f"lag {r['lag_quarters']}Q"),
                html.Td(f"{r['beta_ols']:.4f}"),
                html.Td(f"±{r['beta_ols_se']:.4f}"),
                html.Td(f"{r['r2']:.4f}"),
                html.Td(str(r.get("n_observations", ""))),
                html.Td(r.get("period_label", "")),
            ])
        )

    return [
        html.H3("Bond-Yield Elasticity (OLS)", style={"margin": "0 0 8px 0", "fontSize": "1rem"}),
        html.P("⚠ Correlación observacional — no interpretar como causal",
               style={"color": COLORS["accent2"], "fontSize": "0.8rem", "margin": "0 0 8px 0"}),
        html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"},
            children=[
                html.Thead(html.Tr([
                    html.Th(h, style={"textAlign": "left", "padding": "6px", "borderBottom": f"1px solid {COLORS['grid']}"})
                    for h in ["Region", "Lag", "β OLS", "SE", "R²", "N obs", "Periodo"]
                ])),
                html.Tbody(rows),
            ],
        ),
    ]


if __name__ == "__main__":
    print("\n  Dashboard → http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
