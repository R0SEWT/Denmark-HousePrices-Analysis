# Verification — Plan end-to-end

## Prerequisitos

```bash
source /shared/Code/hackathon-participants/.venv/bin/activate
cd /shared/Code/Denmark-HousePrices-Analysis
git switch robustness-feature-engineering
```

## Fase 1: Verificación estructural (después de cada tarea)

Cada `tasks/NN-*.md` define sus propios criterios de aceptación bajo "Cómo verificar". Ejecutar esa sección antes de marcar `status: done`.

## Fase 2: Verificación integrada (tras completar RFE-07)

Reejecutar `notebooks/03_feature_engineering.ipynb` end-to-end. Condiciones:

1. **Columnas finales no contienen**: `purchase_price`, `price_per_sqm`, `price_zscore`, `price_category`, `price_per_sqm_x_region`, `sqm_x_region`, `regional_price_mean`, `regional_price_median`, `regional_price_std`, `regional_price_cv`, `regional_price_rank`, `regional_liquidity_score`, `regional_p90`, `regional_median`, `region_target_encoded`.
2. **Columnas finales SÍ contienen**: `log_price` (target), `rolling_regional_mean`, `rolling_regional_median`, `rolling_regional_std`, `rolling_regional_cv`, `rolling_regional_count`, `rolling_regional_p90`, `is_premium_causal`, `price_deviation_from_rolling_median`.
3. **Correlation gate**:
   ```python
   corr = df.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
   top = corr.drop(TARGET).head(10)
   assert (top < 0.90).all(), f"Correlación sospechosa: {top}"
   ```
4. **Causalidad de la ventana**: sanity test sintético en RFE-09.

## Fase 3: Verificación de modelado (tras completar RFE-08)

Reejecutar `notebooks/04_modelado_supervisado.ipynb`:

1. `load_scaled_data()` imprime n_features esperada (~20–30).
2. Walk-forward CV corre 9 folds; cada fold reporta RMSE > 0 sobre `log_price` (valores típicos en [0.15, 0.45]).
3. Optuna study `h2o_xgb_walkforward_v1` persiste en `h2o_xgb_gpu_tuning2.db`.
4. Evaluación final sobre holdout 2023–2024:
   - R² ∈ [0.70, 0.88]
   - RMSE en DKK (tras `np.expm1`) coherente con distribución de precios en el período.

## Fase 4: Leak test independiente (RFE-09)

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train_sample, y_train_sample)
r2_lr = lr.score(X_test_sample, y_test_sample)
assert r2_lr < 0.95, f"R² de LinearRegression = {r2_lr:.4f} — posible leak residual"
```

## Fase 5: Checklist de PR

- [ ] Todos los `tasks/NN-*.md` con `status: done` en su frontmatter.
- [ ] Notebooks re-ejecutados y sus outputs commiteados.
- [ ] R² holdout reportado en el cuerpo del PR.
- [ ] SHAP top-10 (del notebook 05) NO dominado por `is_premium_causal` o `price_deviation_from_rolling_median`. Si lo están → crear tarea de follow-up RFE-10.
- [ ] PR apunta a `dev`, no a `main`.
- [ ] Descripción del PR enlaza a `doc/plans/robustness-feature-engineering/README.md`.
