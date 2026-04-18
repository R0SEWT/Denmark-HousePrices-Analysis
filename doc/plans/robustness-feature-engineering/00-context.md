# 00 — Contexto y diagnóstico

## Síntoma

El pipeline actual reporta **R² ≈ 0.9999 sobre `log_price`** para XGBoost tuneado con Optuna. Ese valor no es consistente con precio residencial real — sintomático de data leakage.

## Pipeline actual

```
notebook 03 (feature engineering)
  ├── src/feature_engineering.py
  │    ├── create_price_features()        ← emite log_price, price_per_sqm, price_zscore, price_category
  │    ├── create_regional_aggregated_features()  ← mean/median/std/cv/rank globales
  │    ├── create_advanced_features()     ← is_premium, price_deviation_from_median (con p90/median global)
  │    └── prepare_final_dataset()        ← exclude_cols INSUFICIENTE
  └── produce: data/processed/train_data.parquet, test_data.parquet

notebook 04 (modelado)
  ├── load_scaled_data() exclude_features INSUFICIENTE
  └── H2O + Optuna + XGBoost con un único split (1992-2017 / 2018-2024)
```

## Inventario de leaks

### Leaks directos (columnas ≡ función del target)

| Columna | Origen | Severidad |
|---|---|---|
| `purchase_price` | Target crudo; presente en parquet final | **CRÍTICA** |
| `price_per_sqm = purchase_price / sqm` | `src/feature_engineering.py:1110` | **CRÍTICA** |
| `price_category` (quantile bins) | `src/feature_engineering.py:1113-1117` | **CRÍTICA** |
| `price_zscore` | `src/feature_engineering.py:1120` | **CRÍTICA** |
| `is_premium = purchase_price > regional_p90` | `src/feature_engineering.py:1390` | **CRÍTICA** |
| `price_deviation_from_median = purchase_price - regional_median` | `src/feature_engineering.py:1396` | **CRÍTICA** |
| `price_per_sqm_x_region`, `sqm_x_region` | `src/feature_engineering.py:1330-1353` | **CRÍTICA** |

### Leaks por contaminación train↔test (stats globales antes del split)

| Columna | Origen | Severidad |
|---|---|---|
| `regional_price_mean`, `regional_price_median`, `regional_price_std`, `regional_price_cv` | `src/feature_engineering.py:763-774` | Alta |
| `regional_price_rank`, `regional_liquidity_score` | `src/feature_engineering.py:777-780` | Alta |
| `regional_p90`, `regional_median` (usadas en advanced) | `src/feature_engineering.py:1388-1396` | Alta |
| `region_target_encoded` | `src/feature_engineering.py:413-425` | Alta (ya excluido en notebook 04) |

### Exclusiones actuales (insuficientes)

**`src/feature_engineering.py:1420-1427`** — `prepare_final_dataset`:
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
**Deja pasar**: `purchase_price`, `price_per_sqm`, `is_premium`, `price_deviation_from_median`, `regional_price_mean/median/std/cv`, `regional_price_rank`, `regional_liquidity_score`, `price_per_sqm_x_region`, `sqm_x_region`.

**`notebooks/04_modelado_supervisado.ipynb`** celda `48c5a049`:
```python
exclude_features = [TARGET, "quarter", "region_count", "price_deviation_from_median",
                    "time_trend", "region_target_encoded", "region_count"]
```
Mismo problema: no excluye `purchase_price` ni `price_per_sqm` ni `is_premium` ni los agregados regionales.

## Split actual

**Correcto a nivel de filas** (`src/feature_engineering.py:1559-1560`):
```python
train_mask = df['year'] <= 2017
test_mask = df['year'] >= 2018
```

**Pero contaminado a nivel de features**: las agregaciones de arriba se calculan sobre el dataset completo (train+test unidos) antes del split. Para una fila de 1992 su `regional_price_mean` incluye precios de 2024.

## Objetivo del plan

1. Eliminar todas las columnas que son función casi-exacta del target.
2. Reemplazar agregaciones globales por **ventana rolling causal** (`[year-k, year-1]` por región).
3. Introducir **walk-forward expanding CV** para validación honesta durante tuning Optuna.
4. Añadir gates de verificación (correlación, leak test con regresión lineal) que eviten que un leak futuro pase sin detectar.

## R² esperado después del fix

Sobre `log_price`, rango realista **[0.70, 0.88]**. Si queda > 0.95 tras RFE-09, iterar sobre decisiones de `02-decisions.md` (en particular eliminar `is_premium` y `price_deviation_from_median` aun siendo causales).
