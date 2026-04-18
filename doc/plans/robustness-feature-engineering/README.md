# Robustness Feature Engineering — Runbook

Plan para remediar el data leak del modelo XGBoost (R² ≈ 0.9999 → esperado 0.70–0.88 sobre `log_price`) mediante **agregaciones causales con ventana rolling** y **walk-forward expanding CV**.

> **Rama**: `robustness-feature-engineering` (sale de `dev`)
> **Target de merge**: PR → `dev`
> **Estado global**: scaffolding publicado — tareas pendientes de ejecución.

## Índice

| Documento | Propósito |
|---|---|
| [00-context.md](./00-context.md) | Problema, diagnóstico, evidencia de leak |
| [01-scope.md](./01-scope.md) | In-scope / out-of-scope, criterios de éxito |
| [02-decisions.md](./02-decisions.md) | ADR: ventana rolling k=3, walk-forward, is_premium causal |
| [03-addresses.md](./03-addresses.md) | Mapa `file:line` de todos los sitios a modificar |
| [verification.md](./verification.md) | Plan de verificación end-to-end |

## Runbook agentico

Las tareas están en orden topológico. Un agente que entra a esta rama:

1. Lee [00-context.md](./00-context.md) + [02-decisions.md](./02-decisions.md) para contexto.
2. Abre [TaskList](#lista-de-tareas), toma la primera con `status: todo` cuyas `depends_on` estén `done`.
3. Ejecuta los cambios descritos en el task doc, corre la verificación local.
4. Marca el task como `status: done` actualizando el frontmatter YAML del archivo correspondiente.
5. Vuelve al paso 2 hasta completar la rama.
6. Abre PR hacia `dev` cuando `09-validation` quede `done`.

### Lista de tareas

| ID | Título | Archivo | Estado | Depende de |
|---|---|---|---|---|
| RFE-01 | Config knobs de window + CV | [tasks/01-config-knobs.md](./tasks/01-config-knobs.md) | todo | — |
| RFE-02 | `create_rolling_regional_features()` causal | [tasks/02-rolling-features.md](./tasks/02-rolling-features.md) | todo | RFE-01 |
| RFE-03 | Prune price_per_sqm / price_zscore / price_category | [tasks/03-price-features-prune.md](./tasks/03-price-features-prune.md) | todo | — |
| RFE-04 | is_premium / price_deviation con umbral causal | [tasks/04-advanced-causal.md](./tasks/04-advanced-causal.md) | todo | RFE-02 |
| RFE-05 | Endurecer exclude_cols + assert anti-leak | [tasks/05-exclude-hardening.md](./tasks/05-exclude-hardening.md) | todo | RFE-03, RFE-04 |
| RFE-06 | `create_walk_forward_folds()` + loop Optuna | [tasks/06-walk-forward-cv.md](./tasks/06-walk-forward-cv.md) | todo | RFE-01 |
| RFE-07 | Update notebook 03 (FE) | [tasks/07-notebook-03.md](./tasks/07-notebook-03.md) | todo | RFE-02, RFE-03, RFE-04, RFE-05 |
| RFE-08 | Update notebook 04 (modelado) | [tasks/08-notebook-04.md](./tasks/08-notebook-04.md) | todo | RFE-06, RFE-07 |
| RFE-09 | Smoke tests + correlation gate + leak test | [tasks/09-validation.md](./tasks/09-validation.md) | todo | RFE-08 |

## Convenciones

- **Frontmatter YAML**: cada `tasks/NN-*.md` empieza con `---` y campos `id`, `title`, `branch`, `status`, `depends_on`, `touches`, `estimated_loc`.
- **Estados**: `todo` → `in-progress` → `done`. Solo agentes que están ejecutando una tarea pueden ponerla en `in-progress`.
- **Commits por tarea**: prefijar con `[RFE-NN]` en el mensaje de commit, e.g. `[RFE-02] feat(fe): create_rolling_regional_features causal con ventana k años`.
- **Entorno**: `source /shared/Code/hackathon-participants/.venv/bin/activate` antes de cualquier ejecución.
