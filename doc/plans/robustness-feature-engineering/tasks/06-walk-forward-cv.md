---
id: RFE-06
title: create_walk_forward_folds() + loop Optuna
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-01]
touches:
  - src/feature_engineering.py:1585 (append)
estimated_loc: ~60
---

## Objetivo

Añadir utilidad que emite folds walk-forward expanding, consumible por el `objective` de Optuna en notebook 04.

## Contexto

Ver [02-decisions.md §ADR-02](../02-decisions.md). Esquema:
- Fold i: train años [1992, MIN_TRAIN_END + i], validation = año MIN_TRAIN_END + i + 1
- Para MIN=2013, MAX=2021 → 9 folds (val ∈ {2014, ..., 2022})
- Holdout: años ≥ HOLDOUT_START_YEAR (2023)

## Cambios exactos

**Archivo**: `src/feature_engineering.py`, añadir al final (después de `save_feature_engineering_artifacts`, ~línea 1585):

```python
def create_walk_forward_folds(
    df: pd.DataFrame,
    selected_features: List[str],
    target_col: str = 'log_price',
    year_col: str = 'year',
    min_train_end: int = None,
    max_train_end: int = None,
    holdout_start: int = None,
) -> Dict[str, Any]:
    """
    Genera folds walk-forward expanding + holdout final.

    Returns dict con:
      - folds: List[Tuple[X_train, y_train, X_val, y_val, fold_meta]]
      - holdout: Tuple[X_train_full, y_train_full, X_holdout, y_holdout]
      - config: dict con años usados
    """
    from config import CV_MIN_TRAIN_END_YEAR, CV_MAX_TRAIN_END_YEAR, HOLDOUT_START_YEAR

    if min_train_end is None:
        min_train_end = CV_MIN_TRAIN_END_YEAR
    if max_train_end is None:
        max_train_end = CV_MAX_TRAIN_END_YEAR
    if holdout_start is None:
        holdout_start = HOLDOUT_START_YEAR

    if year_col not in df.columns:
        raise ValueError(f"Columna '{year_col}' requerida para walk-forward")

    folds = []
    for train_end in range(min_train_end, max_train_end + 1):
        val_year = train_end + 1
        if val_year >= holdout_start:
            break
        train_mask = df[year_col] <= train_end
        val_mask = df[year_col] == val_year
        X_train = df.loc[train_mask, selected_features].copy()
        y_train = df.loc[train_mask, target_col].copy()
        X_val = df.loc[val_mask, selected_features].copy()
        y_val = df.loc[val_mask, target_col].copy()
        folds.append({
            'train_end': train_end,
            'val_year': val_year,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
        })
        print(f"  fold train≤{train_end} (n={len(X_train):,}) / val={val_year} (n={len(X_val):,})")

    # Holdout final: train = todos los años < holdout_start; holdout = años >= holdout_start
    train_full_mask = df[year_col] < holdout_start
    holdout_mask = df[year_col] >= holdout_start
    holdout = {
        'X_train_full': df.loc[train_full_mask, selected_features].copy(),
        'y_train_full': df.loc[train_full_mask, target_col].copy(),
        'X_holdout': df.loc[holdout_mask, selected_features].copy(),
        'y_holdout': df.loc[holdout_mask, target_col].copy(),
        'holdout_years': sorted(df.loc[holdout_mask, year_col].unique().tolist()),
    }

    print(f"📊 {len(folds)} folds walk-forward + holdout [{holdout['holdout_years']}]")

    return {
        'folds': folds,
        'holdout': holdout,
        'config': {
            'min_train_end': min_train_end,
            'max_train_end': max_train_end,
            'holdout_start': holdout_start,
            'n_folds': len(folds),
        },
    }
```

### Loop Optuna (para consumir en notebook 04)

Snippet de referencia que irá en RFE-08:

```python
import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator
from feature_engineering import create_walk_forward_folds
from config import OPTUNA_STUDY_NAME_V2

fold_data = create_walk_forward_folds(df_full, selected_features, target_col=TARGET)

def objective(trial):
    params = {
        'ntrees': trial.suggest_int('ntrees', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learn_rate': trial.suggest_float('learn_rate', 0.005, 0.3, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'sample_rate': trial.suggest_float('sample_rate', 0.5, 1.0),
        'col_sample_rate': trial.suggest_float('col_sample_rate', 0.5, 1.0),
        'backend': 'gpu',
    }
    rmses = []
    for fold in fold_data['folds']:
        tr = pd.concat([fold['X_train'], fold['y_train']], axis=1)
        va = pd.concat([fold['X_val'], fold['y_val']], axis=1)
        tr_h = h2o.H2OFrame(tr); va_h = h2o.H2OFrame(va)
        model = H2OXGBoostEstimator(**params)
        model.train(x=selected_features, y=TARGET,
                    training_frame=tr_h, validation_frame=va_h)
        rmses.append(model.rmse(valid=True))
        h2o.remove(tr_h); h2o.remove(va_h)
    return float(np.mean(rmses))

study = optuna.create_study(
    direction='minimize',
    study_name=OPTUNA_STUDY_NAME_V2,
    storage=f"sqlite:///h2o_xgb_gpu_tuning2.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=30)
```

## Criterios de aceptación

- [ ] `create_walk_forward_folds(df_sintético, ['sqm'])` con df que cubre 2000-2024 emite 9 folds (o el número esperado según config) + holdout con años 2023-2024.
- [ ] `fold['X_train']` de fold i y `fold['X_train']` de fold i+1 tienen tamaños crecientes (expanding).
- [ ] `fold['X_val']` contiene exactamente un año.
- [ ] Holdout no solapa con ningún fold de training.

## Cómo verificar

```python
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from feature_engineering import create_walk_forward_folds

# df sintético 2000-2024, 100 obs/año
df = pd.DataFrame({
    'year': np.repeat(range(2000, 2025), 100),
    'sqm': np.random.randint(50, 200, 2500),
    'log_price': np.random.rand(2500) * 2 + 12,
})
out = create_walk_forward_folds(df, ['sqm'],
                                min_train_end=2013, max_train_end=2021, holdout_start=2023)
assert len(out['folds']) == 9
assert out['folds'][0]['train_end'] == 2013 and out['folds'][0]['val_year'] == 2014
assert out['folds'][-1]['train_end'] == 2021 and out['folds'][-1]['val_year'] == 2022
assert 2023 in out['holdout']['holdout_years'] and 2024 in out['holdout']['holdout_years']
# expanding: train crece
sizes = [f['train_size'] for f in out['folds']]
assert all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1))
print("RFE-06 OK")
```
