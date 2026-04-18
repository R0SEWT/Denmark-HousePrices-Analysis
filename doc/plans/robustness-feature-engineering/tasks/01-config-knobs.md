---
id: RFE-01
title: Añadir constantes de window rolling y walk-forward CV a src/config.py
branch: robustness-feature-engineering
status: todo
depends_on: []
touches:
  - src/config.py:153
estimated_loc: ~12
---

## Objetivo

Exponer las constantes de ventana rolling y walk-forward como knobs configurables en `src/config.py`, para que todas las tareas posteriores (RFE-02, RFE-04, RFE-06) las importen desde un único lugar.

## Contexto

Ver [02-decisions.md §ADR-01 y §ADR-02](../02-decisions.md). Decisiones: k=3, MIN_OBS=20, folds walk-forward entre 2013 y 2021, holdout 2023–2024.

## Cambios exactos

**Archivo**: `src/config.py`

Añadir al final del archivo (después de `EVALUATION_METRICS`, línea ~152):

```python
# ============================================
# FEATURE ENGINEERING TEMPORAL (RFE-01)
# Ventana rolling causal para agregaciones regionales
# ============================================

ROLLING_WINDOW_YEARS = 3
"""Número de años previos [year - k, year - 1] para agregaciones regionales causales."""

MIN_OBS_PER_WINDOW = 20
"""Mínimo de observaciones en la ventana para emitir valor; si no, NaN → imputación."""

# ============================================
# WALK-FORWARD CROSS-VALIDATION (RFE-01)
# Folds: train [1992, Y] / val [Y+1] para Y en [MIN, MAX]
# Holdout final: años >= HOLDOUT_START_YEAR
# ============================================

CV_MIN_TRAIN_END_YEAR = 2013
CV_MAX_TRAIN_END_YEAR = 2021
HOLDOUT_START_YEAR = 2023

OPTUNA_STUDY_NAME_V2 = "h2o_xgb_walkforward_v1"
"""Nombre del study Optuna posterior al fix de leak. Convive con el study antiguo en h2o_xgb_gpu_tuning2.db."""
```

## Criterios de aceptación

- [ ] `from config import ROLLING_WINDOW_YEARS, MIN_OBS_PER_WINDOW, CV_MIN_TRAIN_END_YEAR, CV_MAX_TRAIN_END_YEAR, HOLDOUT_START_YEAR, OPTUNA_STUDY_NAME_V2` funciona desde un notebook con `set_project_root()`.
- [ ] `ROLLING_WINDOW_YEARS == 3`, `HOLDOUT_START_YEAR == 2023`, `OPTUNA_STUDY_NAME_V2 == "h2o_xgb_walkforward_v1"`.
- [ ] Ninguna constante preexistente de `config.py` fue alterada.

## Cómo verificar

```bash
source /shared/Code/hackathon-participants/.venv/bin/activate
cd /shared/Code/Denmark-HousePrices-Analysis
python -c "
import sys; sys.path.insert(0, 'src')
import config
assert config.ROLLING_WINDOW_YEARS == 3
assert config.CV_MIN_TRAIN_END_YEAR == 2013
assert config.CV_MAX_TRAIN_END_YEAR == 2021
assert config.HOLDOUT_START_YEAR == 2023
assert config.OPTUNA_STUDY_NAME_V2 == 'h2o_xgb_walkforward_v1'
print('RFE-01 OK')
"
```
