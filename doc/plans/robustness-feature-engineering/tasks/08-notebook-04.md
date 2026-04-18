---
id: RFE-08
title: Actualizar notebook 04 (modelado) — walk-forward CV + holdout
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-06, RFE-07]
touches:
  - notebooks/04_modelado_supervisado.ipynb
estimated_loc: ~varias celdas
---

## Objetivo

Rehacer la celda de Optuna para usar walk-forward CV, y añadir celda de evaluación final sobre holdout 2023–2024.

## Contexto

Ver [06-walk-forward-cv.md](./06-walk-forward-cv.md) para el esqueleto del loop. Decisiones en [02-decisions.md §ADR-02](../02-decisions.md).

## Cambios exactos

### 1. `load_scaled_data()` (celda `48c5a049`, ~línea 1004)

**Antes**:
```python
exclude_features = [TARGET, "quarter", "region_count", "price_deviation_from_median",
                    "time_trend", "region_target_encoded", "region_count"]
```

**Después**:
```python
# RFE-08: lista endurecida (duplica la de src/feature_engineering.py:prepare_final_dataset
# como segunda línea de defensa; si el parquet trae residuos legacy, los atrapamos aquí)
exclude_features = [
    TARGET, 'purchase_price',
    'quarter', 'region_count', 'time_trend',
    'price_per_sqm', 'price_zscore', 'price_category',
    'sqm_x_region', 'price_per_sqm_x_region',
    'is_premium', 'price_deviation_from_median',
    'regional_p90', 'regional_median',
    'regional_price_mean', 'regional_price_median', 'regional_price_std',
    'regional_price_cv', 'regional_price_rank',
    'regional_transaction_count', 'regional_liquidity_score',
    'region_target_encoded',
    'rolling_regional_median_v2',
]
```

### 2. Cargar dataset con columna `year` preservada

**Antes**: `load_scaled_data()` carga `train_data.parquet` + `test_data.parquet` (sin year).

**Después**: cargar `processed_data.parquet` (contiene `year`); el split lo hace `create_walk_forward_folds`. Dos opciones:
- (a) Modificar `load_scaled_data()` para retornar `df_full` con `year` además de `X_train/X_test/y_train/y_test`.
- (b) Añadir `load_full_data()` nueva que retorna `(df_full, selected_features, scaler)`.

Elegir **(b)** para no romper celdas posteriores que dependen de la firma vieja.

### 3. Reemplazar celda del `objective` de Optuna

Ver snippet completo en [06-walk-forward-cv.md §Loop Optuna](./06-walk-forward-cv.md). Puntos clave:

- Usa `OPTUNA_STUDY_NAME_V2 = "h2o_xgb_walkforward_v1"` (nuevo study, no sobrescribe el viejo).
- `n_trials` reducido a 30 (9 folds × 30 = 270 training runs totales; mantener viable en GPU).
- Después de cada fold hacer `h2o.remove(tr_h); h2o.remove(va_h)` para liberar memoria.
- Reportar `trial.set_user_attr('rmse_per_fold', rmses)` para inspección post-hoc.

### 4. Nueva celda: Evaluación final sobre holdout

```python
# RFE-08: Evaluación final con mejores hyperparams
best_params = study.best_params
print("Best params:", best_params)
print("Best CV RMSE (mean):", study.best_value)

holdout = fold_data['holdout']
tr_full = pd.concat([holdout['X_train_full'], holdout['y_train_full']], axis=1)
ho = pd.concat([holdout['X_holdout'], holdout['y_holdout']], axis=1)
tr_h = h2o.H2OFrame(tr_full)
ho_h = h2o.H2OFrame(ho)

final_model = H2OXGBoostEstimator(**best_params, backend='gpu')
final_model.train(x=selected_features, y=TARGET,
                  training_frame=tr_h, validation_frame=ho_h)

pred = final_model.predict(ho_h).as_data_frame()['predict'].values
y_true = holdout['y_holdout'].values

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse_log = mean_squared_error(y_true, pred, squared=False)
mae_log = mean_absolute_error(y_true, pred)
r2_log = r2_score(y_true, pred)

# Convertir a DKK
pred_dkk = np.expm1(pred)
true_dkk = np.expm1(y_true)
rmse_dkk = mean_squared_error(true_dkk, pred_dkk, squared=False)
mae_dkk = mean_absolute_error(true_dkk, pred_dkk)

print(f"=== HOLDOUT {holdout['holdout_years']} ===")
print(f"R² (log_price): {r2_log:.4f}")
print(f"RMSE (log_price): {rmse_log:.4f}   MAE (log_price): {mae_log:.4f}")
print(f"RMSE (DKK): {rmse_dkk:,.0f}   MAE (DKK): {mae_dkk:,.0f}")

# Persistir
final_model.model_id
h2o.save_model(final_model, path='models/', force=True)
```

### 5. Eliminar celdas obsoletas (opcional)

Si el notebook tiene celdas de "modelo inicial sin tuning" o "comparación de modelos" con el split único, dejarlas pero marcarlas como referencia histórica con comentario `# LEGACY (split único) — referencia, no usar para reporte final`.

## Criterios de aceptación

- [ ] Notebook corre end-to-end sin errores.
- [ ] Optuna crea study `h2o_xgb_walkforward_v1` en `h2o_xgb_gpu_tuning2.db`.
- [ ] Mejor CV RMSE mostrado > 0 (valor típico 0.15–0.40 sobre `log_price`).
- [ ] Holdout R² ∈ [0.70, 0.88] sobre `log_price`.
- [ ] Si R² > 0.95 → abrir issue/tarea RFE-10 para revisar `is_premium_causal`/`price_deviation_from_rolling_median`.
- [ ] Modelo final guardado en `models/`.

## Cómo verificar

Ejecutar el notebook. Al final, correr:

```python
import optuna
study = optuna.load_study(study_name='h2o_xgb_walkforward_v1',
                          storage='sqlite:///h2o_xgb_gpu_tuning2.db')
print(f"n_trials: {len(study.trials)}")
print(f"Best value (mean RMSE): {study.best_value:.4f}")
assert len(study.trials) > 0
assert 0 < study.best_value < 1.0  # RMSE on log scale
print("RFE-08 OK")
```
