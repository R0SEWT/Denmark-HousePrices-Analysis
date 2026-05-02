---
id: RFE-04
title: is_premium y price_deviation con umbral causal (rolling)
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-02]
touches:
  - src/feature_engineering.py:1383-1401
estimated_loc: ~40
---

## Objetivo

Reemplazar `regional_p90` / `regional_median` (globales) por versiones causales basadas en la misma ventana rolling que `create_rolling_regional_features`, y reemitir `is_premium_causal` y `price_deviation_from_rolling_median`.

## Contexto

Ver [02-decisions.md §ADR-04](../02-decisions.md). Decisión: conservar estas features pero con umbral causal. Si dominan SHAP en RFE-09 → eliminar en iteración futura.

## Cambios exactos

**Archivo**: `src/feature_engineering.py:1383-1401`

Reemplazar el bloque `=== VARIABLES GEOGRÁFICAS AVANZADAS ===` actual:

```python
# RFE-04: umbral causal (rolling window)
from config import ROLLING_WINDOW_YEARS

if 'region' in df_advanced.columns and 'year' in df_advanced.columns:
    # Yearly p90 y median por region
    yearly_p = (df_advanced.groupby(['region', 'year'])[target_col]
                  .agg([('y_p90', lambda s: s.quantile(0.9)),
                        ('y_median', 'median'),
                        ('y_count', 'count')])
                  .reset_index())

    records = []
    for region, grp in yearly_p.groupby('region'):
        grp = grp.sort_values('year').reset_index(drop=True)
        years = grp['year'].values
        p90s = grp['y_p90'].values
        medians = grp['y_median'].values
        counts = grp['y_count'].values
        for i, y_ref in enumerate(years):
            mask = (years >= y_ref - ROLLING_WINDOW_YEARS) & (years <= y_ref - 1)
            if mask.sum() == 0 or counts[mask].sum() < 20:
                rp90 = rmed = np.nan
            else:
                w = counts[mask]
                # p90 ponderada por count (aprox): promedio ponderado
                rp90 = np.average(p90s[mask], weights=w)
                rmed = np.average(medians[mask], weights=w)
            records.append({'region': region, 'year': int(y_ref),
                            'rolling_regional_p90': rp90,
                            'rolling_regional_median_v2': rmed})

    causal_thresh = pd.DataFrame.from_records(records)
    df_advanced = df_advanced.merge(causal_thresh, on=['region', 'year'], how='left')

    df_advanced['is_premium_causal'] = (
        df_advanced[target_col] > df_advanced['rolling_regional_p90']
    ).astype(int)
    print("✅ is_premium_causal")

    df_advanced['price_deviation_from_rolling_median'] = (
        df_advanced[target_col] - df_advanced['rolling_regional_median_v2']
    )
    print("✅ price_deviation_from_rolling_median")
```

Notas:
- `rolling_regional_median_v2` se renombra para no chocar con la columna ya emitida por `create_rolling_regional_features`. Ambas pueden quedar en el df; una será usada como feature y la otra solo como intermedia.
- El umbral `20` para `min_obs` puede importarse de config (`MIN_OBS_PER_WINDOW`) para consistencia; actualizar el código para hacerlo.
- Eliminar las columnas auxiliares `regional_p90` y `regional_median` (globales, legacy) antes de retornar si aún existen.

## Criterios de aceptación

- [ ] Columna `is_premium_causal` existe; `is_premium` (legacy) no.
- [ ] Columna `price_deviation_from_rolling_median` existe; `price_deviation_from_median` y `regional_median` (legacy) no.
- [ ] Para filas de los primeros k años del dataset, ambas columnas son NaN (imputadas downstream en RFE-05).
- [ ] Test causal sintético: si se inyecta un outlier gigante en año Y y región R, las filas de (Y, R) mantienen `rolling_regional_p90` cercano al pre-outlier (historia causal).

## Cómo verificar

```python
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from feature_engineering import create_advanced_features

# Synthetic con outlier en 2005-A
rng = np.random.default_rng(0)
rows = []
for region in ['A', 'B']:
    for year in range(2000, 2010):
        for _ in range(50):
            base = rng.normal(100, 5)
            if year == 2005 and region == 'A':
                base += 100_000   # outlier masivo
            rows.append({'region': region, 'year': year, 'purchase_price': base, 'sqm': 50})
df = pd.DataFrame(rows)

out = create_advanced_features(df, target_col='purchase_price')
row_2005_A = out.query("region == 'A' and year == 2005").iloc[0]
# El umbral rolling p90 viene de 2002-2004 (pre-outlier)
assert row_2005_A['rolling_regional_p90'] < 200, "is_premium_causal contaminado por año actual"
assert 'is_premium' not in out.columns, "Versión legacy todavía presente"
print("RFE-04 OK")
```
