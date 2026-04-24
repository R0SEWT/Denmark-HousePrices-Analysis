# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Academic data-science project (Spanish-language, UPC course *Data Visualization*) analyzing Danish residential house prices 1992–2024 (~1.5M rows, Kaggle dataset by Martin Frederiksen). Two parallel deliverables live in this repo:

1. **Modeling pipeline** — Jupyter notebooks + `src/` library. Final model is XGBoost on a distributed H2O cluster with GPU, tuned with Optuna (R² ≈ 0.9999 on `log_price`).
2. **Data-viz product** (course final project) — Tableau dashboard + Python notebooks + base preparada, evaluated across 7 entregas parciales. The PPT/defensa is authored **outside this repo**.

Notebooks, comments, docstrings, and printed output are in Spanish — preserve that when editing.

## Active branches and division of work

- `data-viz` (current umbrella branch, off `dev`) — only scaffolds plans and aggregates PRs from `viz/hito-NN-<slug>` sub-branches. **Do not put implementation directly on `data-viz`.** See `doc/plans/data-viz/` (README, 00-context, 01-milestones, 02-branching-convention) for hito list and merge policy.
- A separate branch handles the **data-leak fix** in feature engineering / split logic. Do not attempt the leak fix from `data-viz`; coordinate via `dev`.
- `doc/proyect/requerimiento-data-viz.md` = course requirement (what the final product must contain: 7 entregas, min 8 hojas Tableau, 3 bloques funcionales, vista temporal + transversal + componente avanzado).
- `doc/proyect/desarrollo/E1.md` = current written deliverable (Entrega 1: propuesta, pregunta analítica, usuario objetivo, hipótesis, riesgos). Updates for E1 go here, not in notebooks.

## Notebook workflow

Canonical pipeline under `notebooks/`, meant to run in order and share intermediate artifacts via `data/processed/`:

```
00_extraccion_de_datos     → kaggle download → data/raw/
01_exploracion_eda         → EDA on raw data
02_analisis_descriptivo    → univariate/bivariate descriptive stats
03_feature_engineering     → processed_data.parquet, train/test splits, scalers.pkl, feature_metadata.json
04_modelado_supervisado    → H2O + Optuna + XGBoost (GPU)
05_resultados_finales      → evaluation, SHAP, comparison tables
```

Every notebook starts with:
```python
from setup import set_project_root
set_project_root()
```
`notebooks/setup.py` chdirs to project root and appends `src/` to `sys.path`, so notebooks import as `from analysis import ...` or `import config` (not `from src.analysis`). Keep this pattern for new notebooks.

`environment.yml` and `run_all.sh` are empty placeholders — not authoritative.

Shared venv (per data-viz plan): `source /shared/Code/hackathon-participants/.venv/bin/activate`.

## Code layout (`src/`)

Two sub-packages with different roles:

- **`src/analysis/`** — EDA/visualization helpers. `src/__init__.py` does `from .utils import *` / `from .analysis import *`, and `src/utils.py` re-exports from every submodule. This is a facade: call `utils.run_complete_analysis(df)` / `utils.enhanced_univariate_analysis(df, col, kind)` without caring which submodule defines what. When adding an EDA function, put it in the right submodule (`data_quality`, `univariate_analysis`, `enhanced_analysis`, `visualization`, `summary_analysis`) and export it so the star-import in `src/utils.py` picks it up.
- **`src/features/`** — feature-engineering building blocks (`temporal_features`, `categorical_features`, `geospatial_features`, `derived_features`). Imported explicitly from the FE notebook; **not** re-exported through `src/utils.py`.

`src/descriptive_analysis.py` and `src/feature_engineering.py` at the top level are large legacy modules that predate `analysis/` and `features/`; prefer the modular versions for new work.

## `src/config.py` — watch outs

Single source of truth for paths, target column, and training hyperparameters.

- `TARGET` is defined twice — first as `"purchase_price"` (raw), then overwritten to `"log_price"` (log-transformed target used from notebook 03 onward). Reference the raw target explicitly rather than reading `TARGET` if you need DKK.
- Paths come in local (`DATA_DIR = PROJECT_ROOT/"data"`) and distributed (`DISTRIBUTED_DIR = /mnt/sambashare/BigData-DATA/data`) variants. `ISDISTRIBUTED=True` by default selects the Samba-mounted path (how the two-node H2O cluster shares data). On a single machine without the mount, switch to local paths or set `ISDISTRIBUTED=False`.

## Modeling conventions

- **Temporal split**: 1992–2017 train / 2018–2024 test. Never shuffle — it leaks future data.
- **Dropped before training** (leakage or redundancy): `quarter`, `region_count`, `time_trend`, `region_target_encoded`. H2O also auto-drops `phase_covid_era` as constant on train. If you add engineered features, check they don't reintroduce leakage through the target.
- **Scale**: RMSE/MAE are reported on `log_price`, not DKK. Convert with `np.exp` / `np.expm1` before communicating to stakeholders or showing on viz.
- H2O cluster: `http://localhost:54321`, GPU enabled (`USE_GPU=True`). Optuna state persists in `h2o_xgb_gpu_tuning2.db` (SQLite, tracked); deleting restarts tuning.

## Data-viz conventions (when working under `viz/hito-NN-*`)

- Consume the parquet produced by notebook 03; do **not** reprocess raw data per hito.
- Outputs: PNGs/SVGs to `results/charts/`, tables to `results/tablas/`. Both dirs are gitignored — final assets referenced from `doc/memoria/` must be copied in explicitly.
- Do **not** reintroduce leaks: if a viz needs `purchase_price` raw, load it only in that cell; don't propagate into the pipeline.
- Accessible palettes (avoid red/green; prefer `viridis` or `cividis`). Titles/axes/legends in Spanish.
- For a new hito: branch `viz/hito-NN-<slug>` off `data-viz`, scaffold `doc/plans/data-viz/hitos/NN-<slug>/` (README, 00-context, 01-decisions, tasks/, verification.md), commit prefix `[VIZ-NN]`, PR back to `data-viz` (not `dev`).

## Gitignored vs tracked

Everything data-shaped is ignored: `data/`, `models/`, `results/`, `tests/`, plus globally `*.parquet`, `*.csv`, `*.pkl`, `*.h5`. Tracked: notebooks, `src/`, `doc/`, and the Optuna SQLite DB. `.claudeignore` also hides `doc/memoria/`. Assume reviewers will not have outputs locally — regenerate by running notebooks in order.

## Note on existing `CLADE.MD`

A file named `CLADE.MD` (typo) already exists at repo root with earlier guidance. This `CLAUDE.md` supersedes it; the old file can be removed.
