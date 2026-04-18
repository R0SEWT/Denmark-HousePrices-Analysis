---
id: RFE-02
title: Implementar create_rolling_regional_features() causal
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-01]
touches:
  - src/feature_engineering.py:750-784
estimated_loc: ~80
---

## Objetivo

Reemplazar `create_regional_aggregated_features()` por una versión que calcula agregados regionales usando solo datos de años estrictamente anteriores a la fila (ventana `[year - k, year - 1]`).

## Contexto

Ver [00-context.md §Leaks por contaminación train↔test](../00-context.md) y [02-decisions.md §ADR-01](../02-decisions.md). La versión actual hace `groupby(region)[price].agg(...)` sobre todo el df, lo que contamina filas de 1992 con precios de 2024.

## Cambios exactos

**Archivo**: `src/feature_engineering.py`

1. **Conservar** la función vieja `create_regional_aggregated_features` por compatibilidad; añadir docstring deprecation warning al inicio:
   ```python
   def create_regional_aggregated_features(df, ...):
       """DEPRECADO (RFE-02): usa create_rolling_regional_features. Causa contaminación train↔test."""
       import warnings
       warnings.warn("create_regional_aggregated_features es no-causal; usar create_rolling_regional_features", DeprecationWarning, stacklevel=2)
       # ... código existente intacto
   ```

2. **Añadir** nueva función inmediatamente después (antes de la sección de selección de features ~línea 786):

```python
def create_rolling_regional_features(
    df: pd.DataFrame,
    region_col: str = 'region',
    price_col: str = 'purchase_price',
    year_col: str = 'year',
    window_years: int = None,
    min_obs: int = None,
) -> pd.DataFrame:
    """
    Agregados regionales CAUSALES: para cada fila con año y, los estadísticos
    se calculan usando transacciones en la misma región cuyos años ∈ [y - k, y - 1].

    Emite columnas:
      - rolling_regional_mean
      - rolling_regional_median
      - rolling_regional_std
      - rolling_regional_cv
      - rolling_regional_count

    Filas con < min_obs en su ventana → NaN (se imputan downstream por fold).
    """
    from config import ROLLING_WINDOW_YEARS, MIN_OBS_PER_WINDOW
    if window_years is None:
        window_years = ROLLING_WINDOW_YEARS
    if min_obs is None:
        min_obs = MIN_OBS_PER_WINDOW

    print(f"🔄 create_rolling_regional_features: ventana {window_years}a, min_obs={min_obs}")

    out = df.copy()

    # 1) Agregados yearly por region: mean, median, std, count, sum, sumsq
    yearly = (
        df.groupby([region_col, year_col])[price_col]
          .agg(['mean', 'median', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'y_mean', 'median': 'y_median',
                           'std': 'y_std', 'count': 'y_count'})
    )

    # 2) Para cada (region, year_ref), calcular agregados combinando años [y - k, y - 1].
    #    Implementación: sort por region, year; rolling por region con window = k años
    #    usando DatetimeIndex trucado o, alternativa, loop simple: k es pequeño (3).
    records = []
    for region, grp in yearly.groupby(region_col):
        grp = grp.sort_values(year_col).reset_index(drop=True)
        years_idx = grp[year_col].values
        means = grp['y_mean'].values
        medians = grp['y_median'].values
        stds = grp['y_std'].values
        counts = grp['y_count'].values
        for i, y_ref in enumerate(years_idx):
            mask = (years_idx >= y_ref - window_years) & (years_idx <= y_ref - 1)
            if mask.sum() == 0:
                agg_mean = agg_median = agg_std = np.nan
                agg_count = 0
            else:
                c = counts[mask]
                total = c.sum()
                if total < min_obs:
                    agg_mean = agg_median = agg_std = np.nan
                    agg_count = total
                else:
                    # weighted mean por count
                    agg_mean = np.average(means[mask], weights=c)
                    # median approximation: median de medians ponderada no es exacta,
                    # pero es razonable para ventana pequeña
                    agg_median = np.average(medians[mask], weights=c)
                    # std combinada (pooled std simple)
                    agg_std = np.sqrt(np.average(stds[mask] ** 2, weights=c))
                    agg_count = total
            records.append({
                region_col: region,
                year_col: int(y_ref),
                'rolling_regional_mean': agg_mean,
                'rolling_regional_median': agg_median,
                'rolling_regional_std': agg_std,
                'rolling_regional_count': agg_count,
            })

    rolling_df = pd.DataFrame.from_records(records)
    rolling_df['rolling_regional_cv'] = (
        rolling_df['rolling_regional_std'] / rolling_df['rolling_regional_mean']
    )

    out = out.merge(rolling_df, on=[region_col, year_col], how='left')

    n_nan = out['rolling_regional_mean'].isna().sum()
    print(f"  ✅ Aplicado a {out.shape[0]:,} filas; {n_nan:,} NaN (ventana insuficiente)")
    return out
```

## Criterios de aceptación

- [ ] `from feature_engineering import create_rolling_regional_features` funciona.
- [ ] Para un df sintético con 3 regiones × 10 años × 50 obs/año, la función retorna solo NaN en el primer año (no tiene historial).
- [ ] Sanity causal: para una fila con `year == Y`, ninguna estadística usa filas con `year >= Y`. Verificable con un df sintético donde los precios de año `Y` son outliers masivos: `rolling_regional_mean` de filas en año `Y` debe ser indistinguible del caso sin outliers.
- [ ] `create_regional_aggregated_features` original sigue importable y emite `DeprecationWarning`.

## Cómo verificar

```python
import numpy as np, pandas as pd
from feature_engineering import create_rolling_regional_features

# Synthetic: 2 regiones x 6 años x 50 obs
rng = np.random.default_rng(0)
rows = []
for region in ['A', 'B']:
    for year in range(2000, 2006):
        for _ in range(50):
            price = 100 + (10_000 if year == 2005 and region == 'A' else 0) + rng.normal(0, 5)
            rows.append({'region': region, 'year': year, 'purchase_price': price})
df = pd.DataFrame(rows)

out = create_rolling_regional_features(df, window_years=3, min_obs=10)

# Causal: filas de 2005-A no deben reflejar el outlier masivo (su media rolling viene de 2002-2004)
m_2005_A = out.query("region == 'A' and year == 2005")['rolling_regional_mean'].iloc[0]
assert 80 < m_2005_A < 120, f"Causalidad rota: {m_2005_A}"
# Filas de 2000 deben ser NaN (sin historial)
assert out.query("year == 2000")['rolling_regional_mean'].isna().all()
print("RFE-02 OK")
```
