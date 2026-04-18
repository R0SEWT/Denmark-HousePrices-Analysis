# 01 — Scope

## In-scope

- Modificación de `src/feature_engineering.py`, `src/features/derived_features.py`, `src/config.py`.
- Actualización de `notebooks/03_feature_engineering.ipynb` y `notebooks/04_modelado_supervisado.ipynb`.
- Añadir utilidades de validación (celda de correlación post-FE, leak test con regresión lineal).
- Retuning de Optuna sobre el nuevo esquema walk-forward (study_name nuevo; la DB `h2o_xgb_gpu_tuning2.db` se conserva intacta).

## Out of scope

- Cambios al notebook 05 (resultados finales) más allá de las métricas derivadas del nuevo modelo. El análisis de SHAP queda como verificación post-hoc en RFE-09.
- Cambios al pipeline de extracción (notebook 00) o EDA (notebooks 01, 02).
- Refactor del facade `src/utils.py` / `src/__init__.py` salvo que RFE-07 lo requiera.
- Sustitución de XGBoost por otro modelo. Mantenemos XGBoost en H2O con GPU.
- Cambio del target. Sigue siendo `log_price = log1p(purchase_price)`.
- Eliminar el archivo legacy `src/descriptive_analysis.py` y `src/feature_engineering.py`-como-módulo-monolítico (existe refactor previo en `src/features/` que aún no consume el notebook 03; mantener el estado híbrido actual).

## Criterios de éxito

1. **No leak directo**: ninguna columna en el parquet final ni en `X_train/X_test` es función determinística del target en la misma fila. Verificado por RFE-09 smoke test.
2. **No leak temporal**: toda agregación regional en una fila de año `t` usa solo datos de años estrictamente anteriores (`< t`). Verificado por unit test sintético en RFE-09.
3. **Validación robusta**: Optuna optimiza sobre la media de RMSE de ≥ 5 folds walk-forward. Reporta media ± desv. estándar.
4. **Métricas publicables**: R² holdout 2023–2024 en rango [0.70, 0.88] sobre `log_price`. Convertidas a DKK con `np.expm1` en el notebook 05.
5. **Runbook ejecutable**: un agente que abre esta rama puede completar todas las tareas sin preguntas adicionales al usuario (salvo incidentes).

## Criterios de rechazo (iteración si ocurre)

- R² > 0.95 tras aplicar todos los fixes → revisar `is_premium` y `price_deviation_from_median` (ambas pueden quedar como features legítimas pero sospechosas).
- Caída de features informativas (n_features finales < 15) → ensanchar ventana a k=5 o añadir lags adicionales.
- Walk-forward produce RMSE medio que diverge fuerte del holdout → revisar alineamiento temporal de features.
