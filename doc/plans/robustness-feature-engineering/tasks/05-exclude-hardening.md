---
id: RFE-05
title: Endurecer exclude_cols + assert anti-leak en prepare_final_dataset
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-03, RFE-04]
touches:
  - src/feature_engineering.py:1420-1450
estimated_loc: ~25
---

## Objetivo

Expandir `exclude_cols` de `prepare_final_dataset()` para garantizar que ninguna columna derivada de `purchase_price` llegue a `feature_columns`, y añadir `assert` defensivo que falle loud si alguna columna prohibida se cuela.

## Contexto

Ver [00-context.md §Exclusiones actuales](../00-context.md). Incluso después de RFE-03 y RFE-04, es posible que columnas legacy residan en el parquet intermedio (por ejemplo si RFE-03 solo removió la creación pero no las referencias en merges preexistentes). El assert cubre esa superficie.

## Cambios exactos

**Archivo**: `src/feature_engineering.py:1420-1450`

**Antes**:
```python
exclude_cols = [
    'date', 'region', 'house_id', 'address', 'city', 'area', 'zip_code',
    'house_type', 'sales_type', 'season', 'price_category',
    'rooms_category', 'size_category', 'market_phase',
    'regional_p90', 'regional_median', 'decade_built',
    'year_build', 'price_zscore', 'sqm_price', '%_change_between_offer_and_purchase',
    'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%', 'nom_interest_rate%'
]
```

**Después**:
```python
# RFE-05: lista endurecida. Cualquier derivada directa del target debe listarse aquí.
exclude_cols = [
    # Identificadores y columnas no-feature
    'date', 'region', 'house_id', 'address', 'city', 'area', 'zip_code',
    'house_type', 'sales_type', 'season', 'market_phase',
    'rooms_category', 'size_category', 'decade_built', 'year_build',
    # Target crudo (leak total — log_price = log1p(purchase_price))
    'purchase_price',
    # Derivadas directas del target (eliminadas en RFE-03/04 pero defensivas aquí)
    'price_per_sqm', 'price_zscore', 'price_category',
    'sqm_x_region', 'price_per_sqm_x_region',
    'is_premium', 'price_deviation_from_median',
    'regional_p90', 'regional_median',
    # Agregados regionales globales (legacy, eliminados en RFE-02)
    'regional_price_mean', 'regional_price_median', 'regional_price_std',
    'regional_price_cv', 'regional_price_rank',
    'regional_transaction_count', 'regional_liquidity_score',
    'region_target_encoded', 'region_count',
    # Intermedias auxiliares
    'rolling_regional_median_v2',
    # Macro / variables económicas redundantes
    'sqm_price', '%_change_between_offer_and_purchase',
    'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%', 'nom_interest_rate%',
    # Redundantes temporales
    'quarter', 'time_trend',
]

# === ASSERT ANTI-LEAK (RFE-05) ===
FORBIDDEN = {
    'purchase_price', 'price_per_sqm', 'price_zscore', 'price_category',
    'price_per_sqm_x_region', 'sqm_x_region', 'is_premium',
    'price_deviation_from_median', 'region_target_encoded',
    'regional_price_mean', 'regional_price_median', 'regional_price_std',
    'regional_price_cv', 'regional_price_rank',
}
```

Y tras la línea `feature_columns = [col for col in all_columns if col not in exclude_cols + [target_col]]`:

```python
leaks_detected = FORBIDDEN & set(feature_columns)
assert not leaks_detected, (
    f"RFE-05 anti-leak guard: columnas prohibidas en feature_columns: {leaks_detected}. "
    f"Revisar pipeline upstream o añadir a exclude_cols."
)
```

## Criterios de aceptación

- [ ] `prepare_final_dataset(df_con_leak)` con un df que contiene `purchase_price` falla con `AssertionError` descriptivo.
- [ ] `prepare_final_dataset(df_limpio)` (tras aplicar pipeline RFE-02/03/04) pasa sin error y `feature_columns` no contiene ninguna de `FORBIDDEN`.
- [ ] Contador de features candidatas: ~20–30 (ni < 10 ni > 50).

## Cómo verificar

```python
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from feature_engineering import prepare_final_dataset

# Caso 1: df con leak → debe fallar
df_bad = pd.DataFrame({
    'log_price': np.log1p(np.arange(100) * 1000 + 100),
    'purchase_price': np.arange(100) * 1000 + 100,  # ← leak
    'sqm': np.random.randint(50, 200, 100),
    'no_rooms': np.random.randint(1, 6, 100),
    'month_sin': np.random.rand(100), 'month_cos': np.random.rand(100),
    'quarter_sin': np.random.rand(100), 'quarter_cos': np.random.rand(100),
})
try:
    prepare_final_dataset(df_bad, target_col='log_price')
    print("FAIL: debió lanzar AssertionError")
except AssertionError as e:
    assert "anti-leak" in str(e)
    print("RFE-05 OK (leak detectado correctamente)")
```
