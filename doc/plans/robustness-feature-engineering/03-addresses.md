# 03 — Mapa de direcciones (`file:line`)

Referencias canónicas a todos los sitios del código que las tareas `RFE-NN` van a tocar. Si algún agente detecta que las líneas se corrieron (ediciones previas en la rama), **debe actualizar este documento** antes de continuar.

## src/config.py

| Línea | Qué hay | Acción |
|---|---|---|
| 25, 63 | `TARGET = "purchase_price"` y luego `TARGET = "log_price"` | No tocar — ya documentado en CLAUDE.md |
| 153 (fin de archivo) | — | **Añadir bloque** de constantes window + CV (RFE-01) |

## src/feature_engineering.py

| Línea | Función / bloque | Acción |
|---|---|---|
| 413–425 | `region_target_encoded` (groupby mean smoothed) | Ya excluida en notebook 04; verificar que siga excluida tras RFE-05 |
| 750–784 | `create_regional_aggregated_features` — global | **Reemplazar** por `create_rolling_regional_features` causal (RFE-02) |
| 1095–1124 | `create_price_features` — emite log_price/price_per_sqm/price_zscore/price_category | **Recortar**: solo emite `log_price`; resto eliminado (RFE-03) |
| 1319–1353 | Interacciones `sqm_x_region`, `price_per_sqm_x_region` | **Eliminar** (RFE-03) |
| 1380–1401 | `create_advanced_features` — `regional_p90`, `regional_median`, `is_premium`, `price_deviation_from_median` | **Refactor** a versiones causales con rolling (RFE-04) |
| 1403–1435 | `prepare_final_dataset` — `exclude_cols` | **Endurecer** + añadir assert anti-leak (RFE-05) |
| 1540–1585 | `create_train_test_split` | **No reemplazar** — sigue sirviendo para holdout final. **Añadir** función nueva `create_walk_forward_folds` al lado (RFE-06) |

## src/features/derived_features.py

| Línea | Qué hay | Acción |
|---|---|---|
| 7–18 | `create_price_derived_features` con `price_ratio_regional_median` (mediana global) | **Eliminar** esa función completa; equivalente causal vive en `feature_engineering.py` (RFE-03) |
| 20–26 | `create_size_derived_features` con `sqm_per_room` | Mantener (no leak) |

## notebooks/03_feature_engineering.ipynb

Celdas a actualizar (identificar por ID en JSON del ipynb, no por número de línea):

- Celda que llama a `create_regional_aggregated_features` → reemplazar por `create_rolling_regional_features`.
- Celda que llama a `create_price_features` → ya no emite los 3 extras; solo log_price.
- Celdas que referencian `price_per_sqm`, `price_zscore`, `price_category` en prints o describes → limpiar.
- **Añadir al final**: celda de validación de correlación (RFE-07 / RFE-09).

## notebooks/04_modelado_supervisado.ipynb

Celdas clave:

- Celda `48c5a049` (~línea 1004) — `load_scaled_data()`: **expandir** `exclude_features` (RFE-05 + RFE-08).
- Celda del `objective` de Optuna: **reemplazar** el train/eval único por loop walk-forward (RFE-08).
- Celda de evaluación final: **reentrenar** con mejores hyperparams sobre 1992–2022 y evaluar sobre holdout 2023–2024 (RFE-08).

## Fuera de scope (no tocar)

- `src/analysis/*` — EDA, no FE.
- `src/descriptive_analysis.py` — legacy, no lo consume el pipeline de modelado.
- `h2o_xgb_gpu_tuning2.db` — DB Optuna; el nuevo study se guarda con study_name distinto en la misma DB.
