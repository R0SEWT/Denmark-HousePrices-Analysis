# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Project context

Academic data-science project (Spanish-language) analyzing Danish residential house prices 1992–2024 (~1.5M rows, Kaggle dataset by Martin Frederiksen). Deliverables are Jupyter notebooks plus a supporting `src/` library. Final modeling runs on a distributed H2O cluster with GPU; the chosen model is XGBoost tuned with Optuna (R² ≈ 0.9999 on `log_price`).

Notebooks, comments, docstrings, and printed output are in Spanish — preserve that when editing.

## Unified data flow

Both pipelines share Bronze as the canonical clean layer:

```
data/raw/DKHousingPrices.parquet
  └─► make bronze  → data/bronze/transactions.parquet
        ├─► ML Pipeline (notebook 03): reads Bronze via ML_INPUT_FILE
        │     → feature_engineering → data/processed/ → H2O XGBoost (notebook 04)
        └─► Analytics Pipeline (make silver gold tableau)
              → Silver → Gold (5 KPIs) → Hyper → Tableau Dashboard
```

Run `make all` for the full medallion pipeline. Run `make ml-pipeline` to prepare Bronze for notebook 03.

## Notebook workflow

Two notebook sequences share the same Bronze clean layer:

**ML Pipeline** (notebooks 00–05):
```
00_extraccion_de_datos     → kaggle download → data/raw/
01_exploracion_eda         → EDA on raw data
02_analisis_descriptivo    → univariate/bivariate descriptive stats
03_feature_engineering     → reads Bronze → processed_data.parquet, train/test splits
04_modelado_supervisado    → H2O + Optuna + XGBoost (GPU)
05_resultados_finales      → evaluation, SHAP, comparison tables
```

**Medallion Observability** (notebooks 06–08):
```
06_bronze_quality          → validates Bronze layer output
07_silver_quality          → visualizes Silver enrichments (CPI, geo, STL)
08_gold_kpis               → renders 5 Gold KPIs with macro shock annotations
```

Every notebook starts with:
```python
from setup import set_project_root
set_project_root()
```
`notebooks/setup.py` chdirs to the project root and appends `src/` to `sys.path`, so notebooks import as `from analysis import ...` or `import config` rather than `from src.analysis`. Keep this pattern when adding notebooks.

## Code layout (`src/`)

Three sub-packages:

- **`src/analysis/`** — EDA/visualization helpers. `src/utils.py` re-exports from every submodule (`data_quality`, `univariate_analysis`, `enhanced_analysis`, `visualization`, `summary_analysis`). Downstream code calls `utils.run_complete_analysis(df)` without caring which submodule defines what. When adding an EDA function, put it in the right submodule and export from its top level so the star-import in `src/utils.py` picks it up.
- **`src/medallion/`** — Bronze → Silver → Gold pipeline (`bronze.py`, `silver.py`, `gold.py`, `deflation.py`, `geo.py`, `stl.py`, `metadata.py`). Tested by `tests/test_bronze.py`, `test_silver.py`, `test_gold.py`.
- **`src/tableau/`** — Tableau integration (`hyper_writer.py`, `publish.py`, `workbook.py`). Converts Gold KPIs to `.hyper` extracts and publishes to Tableau Server.
- **`src/features/`** — Only `derived_features.py` remains (size-derived features). Other stubs were removed as dead code.

`src/config.py` is the single source of truth for paths, target column, and training hyperparameters. Key points:

- `TARGET` is defined twice — first as `"purchase_price"` (raw), then overwritten to `"log_price"` (the log-transformed target used by modeling from notebook 03 onward). If you need the raw target, reference it explicitly rather than reading `TARGET`.
- `ML_INPUT_FILE` points to `data/bronze/transactions.parquet` — the canonical clean source for the ML pipeline.
- Paths come in local (`DATA_DIR`) and distributed (`DISTRIBUTED_DIR`) variants. `ISDISTRIBUTED=True` by default selects the Samba-mounted path for the H2O cluster.

`src/descriptive_analysis.py` and `src/feature_engineering.py` are legacy monolith modules used by notebooks 01–03. They work and are frozen — prefer the modular packages for new work.

## Modeling conventions

- **Temporal split**: 1992–2017 train / 2018–2024 test. Never shuffle — doing so leaks future data into training.
- **Variables deliberately dropped before training** (data-leakage or redundancy): `quarter`, `region_count`, `time_trend`, `region_target_encoded`. H2O will also auto-drop `phase_covid_era` as constant on the training window. If you add new engineered features, check they don't reintroduce leakage through the target.
- **Scale**: modeling metrics (RMSE, MAE) are reported on `log_price`, not DKK. Convert back with `np.exp` before communicating to stakeholders.
- H2O cluster is configured for `http://localhost:54321` with GPU enabled (`USE_GPU=True`). Optuna study state is persisted in `h2o_xgb_gpu_tuning2.db` (SQLite) at the repo root — this file is committed and stores tuning history; deleting it restarts Optuna from scratch.

## What's gitignored vs. tracked

Everything data-shaped is ignored: `data/`, `models/`, `results/`, and globally `*.parquet`, `*.csv`, `*.pkl`, `*.h5`. Only the notebooks, `src/`, docs, and the Optuna SQLite DB are tracked. When referencing outputs in documentation or PRs, assume a reviewer will not have them locally — regenerate by running notebooks in order.

## Cost Optimization

- When spawning subagents (Agent tool), set `model: "sonnet"` by default. Only use Opus subagents for tasks that genuinely require frontier reasoning.
- Run `/compact` proactively when context grows large.
