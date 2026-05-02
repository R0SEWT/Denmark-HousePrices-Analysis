# 02 — Decisiones (ADR-style)

Cada decisión tiene: contexto breve, alternativas, elegida, consecuencias. Formato compacto.

---

## ADR-01: Ventana rolling fija de k=3 años (no expanding ni global)

**Contexto**: Las agregaciones regionales actuales (`regional_price_mean`, etc.) se calculan sobre todo el dataset → contaminan train con info del futuro.

**Alternativas**:
- Global (status quo) — leak garantizado.
- Expanding desde 1992 hasta `t-1` — causal, pero estadísticas muy pesadas al pasado (dataset empieza en 1992, mercado post-2008 tiene estructura distinta).
- Rolling fija de k años — causal + responde a cambios de régimen (2008, covid).
- Híbrida rolling+expanding de fallback — más compleja, marginal.

**Elegida**: **Rolling fija con k=3 años** por defecto, constante `ROLLING_WINDOW_YEARS` en `src/config.py`. Filas con menos de `MIN_OBS_PER_WINDOW=20` observaciones en su ventana → NaN → imputadas con mediana global del split de entrenamiento (se recalcula por fold en walk-forward).

**Consecuencias**: las primeras 3 años del dataset (1992–1994) tendrán NaN en agregados regionales → imputados. Aceptable; son < 10% del volumen.

---

## ADR-02: Walk-forward expanding CV (no rolling, no split simple)

**Contexto**: Un único split 1992-2017 / 2018-2024 da un solo estimador de error, susceptible a suerte de la ventana elegida.

**Elegida**: **Walk-forward expanding**. Folds:
- Fold 1: train 1992–2013, val 2014
- Fold 2: train 1992–2014, val 2015
- …
- Fold 9: train 1992–2021, val 2022
- Holdout final: **2023–2024** (nunca visto durante tuning).

9 folds * Optuna trials — coste de cómputo alto pero aceptable con GPU. Configurable via `CV_MIN_TRAIN_END_YEAR=2013`, `CV_MAX_TRAIN_END_YEAR=2021`, `HOLDOUT_START_YEAR=2023` en config.

**Consecuencias**: n_trials de Optuna tendrá que bajar (ej. de 100 → 30) para mantener el tiempo total comparable. El study_name nuevo (`h2o_xgb_walkforward_v1`) convive con la DB antigua en `h2o_xgb_gpu_tuning2.db`.

---

## ADR-03: `log_price` sigue como target; `purchase_price` nunca en features

**Contexto**: `log_price = np.log1p(purchase_price)`. Tenerlas juntas = leak total.

**Elegida**: `log_price` es el target. `purchase_price` se añade explícitamente a `exclude_cols` y a `exclude_features` con assert defensivo.

---

## ADR-04: Conservar `is_premium` y `price_deviation_from_median` con umbral causal

**Contexto**: Estas features comparan el `purchase_price` de la fila actual con un umbral regional. Si el umbral es causal (rolling p90/median de años previos), NO es leak — el umbral no depende del target de la fila.

**Alternativas**:
- Eliminar ambas (conservador).
- Conservar con umbral causal (elegida).

**Consecuencias**: posible que capten ruido espurio. Mitigación: RFE-09 mide importancia; si dominan top-3 de SHAP, se eliminan en una segunda iteración.

---

## ADR-05: Eliminar `price_per_sqm`, `price_zscore`, `price_category`, `price_per_sqm_x_region`, `sqm_x_region`

**Contexto**: Son función exacta o quasi-exacta del target cuando `sqm` también está disponible.

**Elegida**: **Eliminar por completo del pipeline**, no solo excluir. Evita que vuelvan a colarse en una iteración futura.

---

## ADR-06: Mantener XGBoost en H2O con GPU

No hay razón para cambiar el estimador — el leak no es culpa del modelo, sino de las features. Retuning sobre datos saneados debería producir métricas razonables con el mismo backend.

---

## ADR-07: Ubicación docs-as-code = `doc/plans/robustness-feature-engineering/`

**Alternativas**: `docs/`, `utils/`, raíz.

**Elegida**: reusar el ya existente `doc/` (convención del repo) y añadir subcarpeta `plans/`. No crea una nueva convención top-level y deja espacio para futuros planes (`doc/plans/data-viz/`).
