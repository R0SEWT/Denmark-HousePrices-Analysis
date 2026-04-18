---
id: RFE-03
title: Eliminar price_per_sqm / price_zscore / price_category / interacciones
branch: robustness-feature-engineering
status: todo
depends_on: []
touches:
  - src/feature_engineering.py:1095-1124
  - src/feature_engineering.py:1319-1353
  - src/features/derived_features.py:7-18
estimated_loc: ~30 (net negative)
---

## Objetivo

Eliminar del pipeline FE todas las columnas que son funciones casi-determinísticas de `purchase_price` (el target).

## Contexto

Ver [00-context.md §Leaks directos](../00-context.md) y [02-decisions.md §ADR-05](../02-decisions.md).

## Cambios exactos

### 1. `src/feature_engineering.py:1095-1124` — `create_price_features`

**Antes (actual)**:
```python
def create_price_features(df, target_col='purchase_price', sqm_col='sqm'):
    df_price = df.copy()
    df_price['log_price'] = np.log1p(df_price[target_col])
    df_price['price_per_sqm'] = df_price[target_col] / df_price[sqm_col]
    price_quartiles = df_price[target_col].quantile([0.25, 0.5, 0.75])
    df_price['price_category'] = pd.cut(df_price[target_col], ...)
    df_price['price_zscore'] = (df_price[target_col] - df_price[target_col].mean()) / df_price[target_col].std()
    price_vars = ['log_price', 'price_per_sqm', 'price_category', 'price_zscore']
    return df_price
```

**Después**:
```python
def create_price_features(df, target_col='purchase_price'):
    """
    RFE-03: emite SOLO log_price (target). Las columnas price_per_sqm,
    price_category y price_zscore fueron eliminadas por data leakage directo.
    """
    df_price = df.copy()
    df_price['log_price'] = np.log1p(df_price[target_col])
    print("✅ Variables de precio creadas: ['log_price']")
    return df_price
```

Nota: parámetro `sqm_col` eliminado — ya no se usa.

### 2. `src/feature_engineering.py:1319-1353` — Bloque `=== VARIABLES DE INTERACCIÓN ===`

**Antes**: crea `sqm_x_region`, `price_per_sqm_x_region`.

**Después**: eliminar el bloque entero. Reemplazar por comentario:

```python
# RFE-03: bloque VARIABLES DE INTERACCIÓN eliminado — dependía de price_per_sqm
# y region_target_encoded (ambos leaks). Si se necesitan interacciones, recalcular
# con features causales (rolling_regional_mean × sqm, etc.) en una tarea futura.
```

### 3. `src/features/derived_features.py:7-18` — eliminar `create_price_derived_features`

**Antes**:
```python
def create_price_derived_features(df, price_col='purchase_price'):
    df_result = df.copy()
    df_result['log_price'] = np.log1p(df_result[price_col])
    if 'region' in df_result.columns:
        regional_median = df_result.groupby('region')[price_col].transform('median')
        df_result['price_ratio_regional_median'] = df_result[price_col] / regional_median
    return df_result
```

**Después**: eliminar la función completa. Mantener `create_size_derived_features`. Si algún notebook importa `create_price_derived_features`, romper a propósito — la intención es que migre a `create_price_features` + `create_rolling_regional_features`.

## Criterios de aceptación

- [ ] `grep -rn "price_per_sqm\|price_zscore\|price_category\|sqm_x_region\|price_ratio_regional_median" src/` devuelve 0 coincidencias en código (docstrings/comentarios que describan la eliminación son OK).
- [ ] `create_price_features(df)` retorna un df con `log_price` y SIN las otras columnas de precio.
- [ ] `from features.derived_features import create_price_derived_features` falla con `ImportError`.

## Cómo verificar

```bash
source /shared/Code/hackathon-participants/.venv/bin/activate
cd /shared/Code/Denmark-HousePrices-Analysis
# No debe haber referencias en código (comentarios OK si describen la eliminación)
! grep -rn "price_per_sqm\|price_zscore\b\|price_category\b" src/ | grep -v "^.*#" | grep -v "RFE-03\|eliminad" && echo "FALLA: referencias residuales" || echo "OK"
python -c "
import sys; sys.path.insert(0, 'src')
from feature_engineering import create_price_features
import pandas as pd, numpy as np
df = pd.DataFrame({'purchase_price': [100, 200, 300], 'sqm': [50, 100, 150]})
out = create_price_features(df)
assert 'log_price' in out.columns
assert 'price_per_sqm' not in out.columns
assert 'price_zscore' not in out.columns
assert 'price_category' not in out.columns
print('RFE-03 OK')
"
```
