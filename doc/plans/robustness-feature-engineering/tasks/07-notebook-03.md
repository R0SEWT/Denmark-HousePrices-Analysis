---
id: RFE-07
title: Actualizar notebook 03 (feature engineering) al nuevo pipeline
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-02, RFE-03, RFE-04, RFE-05]
touches:
  - notebooks/03_feature_engineering.ipynb
estimated_loc: ~varias celdas
---

## Objetivo

Re-cablear el notebook FE para que use las funciones causales y emita un parquet final saneado y válido para modelado.

## Contexto

Ver [03-addresses.md §notebooks/03_feature_engineering.ipynb](../03-addresses.md).

## Cambios exactos

### 1. Importaciones

En la celda de imports, añadir/ajustar:

```python
from feature_engineering import (
    create_price_features,              # RFE-03: solo emite log_price
    create_rolling_regional_features,   # RFE-02: agregados causales
    create_advanced_features,           # RFE-04: is_premium_causal + price_deviation_from_rolling_median
    prepare_final_dataset,              # RFE-05: exclude_cols endurecido + assert
    create_train_test_split,
)
```

Eliminar imports de `create_regional_aggregated_features` (deprecated) y de `create_price_derived_features` (eliminada en RFE-03).

### 2. Celdas del pipeline

**Reemplazar** la celda que llama a `create_regional_aggregated_features(...)` por:

```python
df = create_rolling_regional_features(df)
```

**Verificar** que la celda que llama a `create_price_features(df)` sigue funcionando con la firma nueva (ya no requiere `sqm_col`).

**Verificar** que la celda que llama a `create_advanced_features(df, target_col='purchase_price')` ahora emite `is_premium_causal` y `price_deviation_from_rolling_median`, no las versiones legacy.

### 3. Nueva celda de validación (post-FE, pre-save)

Insertar antes de guardar el parquet final:

```python
# === RFE-07: Correlation gate post-FE ===
from config import TARGET
numeric = df.select_dtypes(include=[np.number])
corr = numeric.corr()[TARGET].abs().sort_values(ascending=False)
top = corr.drop(TARGET).head(15)
print("Top-15 correlaciones con el target:")
print(top)
MAX_ALLOWED = 0.90
violating = top[top > MAX_ALLOWED]
if len(violating) > 0:
    raise RuntimeError(
        f"Correlación sospechosa (> {MAX_ALLOWED}) — posible leak residual:\n{violating}"
    )
print("✅ Correlation gate passed")
```

### 4. Celda de guardado final

Confirmar que guarda:
- `data/processed/processed_data.parquet` — con columna `year` intacta (requerida por walk-forward del notebook 04)
- `data/processed/train_data.parquet`, `test_data.parquet` — split tradicional conservado (para compatibilidad con notebook 05)
- `scalers.pkl`, `feature_metadata.json`, `selected_features.txt`

## Criterios de aceptación

- [ ] Notebook corre end-to-end sin errores.
- [ ] Parquet final NO contiene: `purchase_price`, `price_per_sqm`, `price_zscore`, `price_category`, `is_premium`, `price_deviation_from_median`, `regional_price_mean`, `regional_price_median`, `regional_price_std`, `regional_price_cv`, `regional_price_rank`, `regional_liquidity_score`, `regional_p90`, `regional_median`, `region_target_encoded`, `sqm_x_region`, `price_per_sqm_x_region`.
- [ ] Parquet final SÍ contiene: `log_price`, `rolling_regional_mean`, `rolling_regional_median`, `rolling_regional_std`, `rolling_regional_cv`, `rolling_regional_count`, `rolling_regional_p90`, `is_premium_causal`, `price_deviation_from_rolling_median`.
- [ ] Correlation gate pasa (ninguna corr > 0.90 salvo target consigo mismo).
- [ ] `feature_metadata.json` refleja la nueva lista de features.

## Cómo verificar

Correr el notebook en orden. Luego:

```python
import pandas as pd
df = pd.read_parquet('data/processed/processed_data.parquet')

FORBIDDEN = {'purchase_price', 'price_per_sqm', 'price_zscore', 'price_category',
             'is_premium', 'price_deviation_from_median', 'regional_price_mean',
             'regional_price_median', 'regional_price_std', 'regional_price_cv',
             'regional_price_rank', 'regional_liquidity_score',
             'regional_p90', 'regional_median', 'region_target_encoded',
             'sqm_x_region', 'price_per_sqm_x_region'}
EXPECTED = {'log_price', 'rolling_regional_mean', 'rolling_regional_median',
            'rolling_regional_std', 'rolling_regional_cv', 'rolling_regional_count',
            'rolling_regional_p90', 'is_premium_causal',
            'price_deviation_from_rolling_median'}

assert not (FORBIDDEN & set(df.columns)), FORBIDDEN & set(df.columns)
assert EXPECTED <= set(df.columns), EXPECTED - set(df.columns)
print("RFE-07 OK")
```
