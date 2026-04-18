---
id: RFE-09
title: Smoke tests + correlation gate + leak test con regresión lineal
branch: robustness-feature-engineering
status: todo
depends_on: [RFE-08]
touches:
  - notebooks/04_modelado_supervisado.ipynb (celda final)
  - (opcional) tests/test_rolling_causal.py
estimated_loc: ~40
---

## Objetivo

Añadir una batería de verificaciones independientes del pipeline que confirmen:

1. Window rolling es estrictamente causal.
2. `FORBIDDEN` del assert en `prepare_final_dataset` es completo.
3. Un modelo lineal sobre las features finales NO alcanza R² sospechoso (> 0.95).
4. SHAP (en notebook 05) no está dominado por `is_premium_causal` / `price_deviation_from_rolling_median`.

## Contexto

Ver [verification.md §Fase 4](../verification.md). Estas pruebas son la última línea de defensa: si en cinco iteraciones futuras alguien re-introduce un leak sutil, RFE-09 debería atraparlo.

## Cambios exactos

### 1. Test sintético de causalidad (nuevo archivo opcional)

**Archivo**: `tests/test_rolling_causal.py` (crear si no existe `tests/`)

```python
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, 'src')
from feature_engineering import create_rolling_regional_features


def test_rolling_is_causal_with_outlier_year():
    """Si inyectamos un outlier masivo en año Y, las filas de Y no deben reflejarlo."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ['A', 'B']:
        for year in range(2000, 2010):
            for _ in range(40):
                p = rng.normal(100, 5)
                if year == 2005 and region == 'A':
                    p += 100_000
                rows.append({'region': region, 'year': year, 'purchase_price': p})
    df = pd.DataFrame(rows)

    out = create_rolling_regional_features(df, window_years=3, min_obs=10)
    row = out.query("region == 'A' and year == 2005").iloc[0]
    # rolling_regional_mean de 2005-A viene de años 2002, 2003, 2004 (sin outlier)
    assert row['rolling_regional_mean'] < 200, (
        f"Causalidad rota: 2005-A rolling_mean = {row['rolling_regional_mean']}"
    )


def test_rolling_nan_first_years():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'region': ['A'] * 100,
        'year': list(range(2000, 2010)) * 10,
        'purchase_price': rng.normal(100, 5, 100),
    })
    out = create_rolling_regional_features(df, window_years=3, min_obs=5)
    assert out.query("year == 2000")['rolling_regional_mean'].isna().all()
```

### 2. Leak test con regresión lineal (celda final del notebook 04)

```python
# === RFE-09: Leak test independiente ===
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sample_idx = np.random.choice(len(holdout['X_train_full']), min(50_000, len(holdout['X_train_full'])), replace=False)
X_s = holdout['X_train_full'].iloc[sample_idx]
y_s = holdout['y_train_full'].iloc[sample_idx]

scaler = StandardScaler()
X_s_sc = scaler.fit_transform(X_s.fillna(X_s.median()))
X_h_sc = scaler.transform(holdout['X_holdout'].fillna(X_s.median()))

lr = LinearRegression().fit(X_s_sc, y_s)
r2_lr = lr.score(X_h_sc, holdout['y_holdout'])
print(f"Leak test (LinearRegression R² en holdout): {r2_lr:.4f}")

if r2_lr > 0.95:
    raise RuntimeError(
        f"LEAK DETECTADO: LinearRegression alcanza R²={r2_lr:.4f} sin tuning. "
        f"Inspeccionar top-correlaciones con el target."
    )
print("✅ RFE-09 leak test passed")
```

### 3. SHAP check (delegado a notebook 05)

Añadir a `notebooks/05_resultados_finales.ipynb` una aserción al final de la sección SHAP:

```python
top_shap_features = shap_importance.head(3)['feature'].tolist()
SUSPECT = {'is_premium_causal', 'price_deviation_from_rolling_median'}
if set(top_shap_features) & SUSPECT:
    print(f"⚠️  Atención: features sospechosas en top-3 SHAP: {set(top_shap_features) & SUSPECT}")
    print("   Considerar eliminarlas en una iteración RFE-10.")
```

## Criterios de aceptación

- [ ] `pytest tests/test_rolling_causal.py` pasa (si se crea).
- [ ] Celda del leak test en notebook 04 corre y reporta R² < 0.95.
- [ ] Celda SHAP del notebook 05 imprime warning si aplica, pero no rompe el run.
- [ ] README.md (runbook) actualizado: RFE-09 en estado `done`.

## Cómo verificar

```bash
source /shared/Code/hackathon-participants/.venv/bin/activate
cd /shared/Code/Denmark-HousePrices-Analysis
pytest tests/test_rolling_causal.py -v  # si se creó el archivo
# Además, el resultado del leak test en notebook 04 debe quedar impreso en el output del ipynb.
```

## Si alguna verificación falla

- **Correlation gate (> 0.90)**: identificar la columna; añadirla a `FORBIDDEN` en `src/feature_engineering.py` y reejecutar RFE-07.
- **Leak test (R² > 0.95)**: regresión lineal puede estar aprovechando una combinación lineal de features. Inspeccionar `np.abs(lr.coef_)`; las features con coef más altos son candidatas a eliminación.
- **SHAP dominado por is_premium_causal**: crear RFE-10 — eliminar `is_premium_causal` del pipeline y re-tunear.
