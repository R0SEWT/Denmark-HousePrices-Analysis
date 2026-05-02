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

## Notebook workflow

The canonical pipeline is the numbered notebook sequence under `notebooks/` — they are meant to run in order and share intermediate artifacts via files in `data/processed/`:

```
00_extraccion_de_datos     → kaggle download → data/raw/
01_exploracion_eda         → EDA on raw data
02_analisis_descriptivo    → univariate/bivariate descriptive stats
03_feature_engineering     → produces processed_data.parquet, train/test splits, scalers.pkl, feature_metadata.json
04_modelado_supervisado    → H2O + Optuna + XGBoost (GPU)
05_resultados_finales      → evaluation, SHAP, comparison tables
```

Every notebook starts with:
```python
from setup import set_project_root
set_project_root()
```
`notebooks/setup.py` chdirs to the project root and appends `src/` to `sys.path`, so notebooks import as `from analysis import ...` or `import config` rather than `from src.analysis`. Keep this pattern when adding notebooks.

`environment.yml` and `run_all.sh` are empty placeholders — do not treat them as authoritative.

## Code layout (`src/`)

Two sub-packages with different roles:

- **`src/analysis/`** — EDA/visualization helpers. `src/__init__.py` does `from .utils import *` and `from .analysis import *`, and `src/utils.py` re-exports from every submodule of `analysis/`. This is a facade pattern: downstream code calls `utils.run_complete_analysis(df)` / `utils.enhanced_univariate_analysis(df, col, kind)` without caring which submodule defines what. When adding an EDA function, put it in the right submodule (`data_quality`, `univariate_analysis`, `enhanced_analysis`, `visualization`, `summary_analysis`) and export it from that submodule's `__all__`/top level so the star-import in `src/utils.py` picks it up.
- **`src/features/`** — feature-engineering building blocks (`temporal_features`, `categorical_features`, `geospatial_features`, `derived_features`). These are imported explicitly from the FE notebook; they are **not** re-exported through `src/utils.py`.

`src/config.py` is the single source of truth for paths, target column, and training hyperparameters. Two things to watch:

- `TARGET` is defined twice — first as `"purchase_price"` (raw), then overwritten to `"log_price"` (the log-transformed target used by modeling from notebook 03 onward). If you need the raw target, reference it explicitly rather than reading `TARGET`.
- Paths come in local (`DATA_DIR = PROJECT_ROOT/"data"`) and distributed (`DISTRIBUTED_DIR = /mnt/sambashare/BigData-DATA/data`) variants. `ISDISTRIBUTED=True` by default selects the Samba-mounted path, which is how the two-node H2O cluster shares data. On a single machine without the mount, switch to the local paths or set `ISDISTRIBUTED=False`.

`src/descriptive_analysis.py` and `src/feature_engineering.py` at the top level are large legacy modules that predate the `analysis/` and `features/` packages; prefer the modular versions for new work.

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
