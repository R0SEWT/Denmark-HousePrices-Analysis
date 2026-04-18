# 02 — Convención de ramas y merge policy

## Naming

- Paraguas: **`data-viz`** (esta rama, sale de `dev`).
- Sub-ramas por hito: **`viz/hito-NN-<slug>`** (todas salen de `data-viz`, no de `dev`).
  - `NN` = número del hito de [01-milestones.md](./01-milestones.md), con ceros a la izquierda (01, 02, …).
  - `<slug>` = kebab-case corto (máx. 3 palabras), p.ej. `geo`, `series-temporales`, `model-comparison`.

## Flujo por hito

```bash
# Empezar un hito
git switch data-viz
git pull
git switch -c viz/hito-01-geo

# Scaffolding del hito (docs-as-code)
mkdir -p doc/plans/data-viz/hitos/01-geo/tasks
# Escribir README.md, 00-context.md, 01-decisions.md, tasks/*.md al estilo RFE

git add doc/plans/data-viz/hitos/01-geo/
git commit -m "[VIZ-01] docs(plans): scaffolding hito geo"
git push -u origin viz/hito-01-geo

# Implementar tareas...
# Al terminar, abrir PR → data-viz (NO a dev)
```

## Estructura interna de un hito

Dentro de la sub-rama, crear:

```
doc/plans/data-viz/hitos/NN-<slug>/
├── README.md
├── 00-context.md
├── 01-decisions.md
├── tasks/
│   ├── 01-*.md
│   └── ...
└── verification.md
```

Misma plantilla que [robustness-feature-engineering](../robustness-feature-engineering/README.md).

## Commits

- Prefijar con `[VIZ-NN]` o `[VIZ-NN.TK]` para tareas específicas:
  - `[VIZ-01] docs(plans): scaffolding hito geo`
  - `[VIZ-01.02] feat(viz): choropleth precio por region`
  - `[VIZ-02] fix(viz): ajuste de escala log en serie temporal`

## Merge policy

### Sub-rama → `data-viz`

- Requiere que todos los `tasks/NN-*.md` del hito tengan `status: done`.
- Requiere notebooks re-ejecutados con outputs actualizados.
- Merge via **PR squash** (una sola commit limpia por hito en `data-viz`).

### `data-viz` → `dev`

- Solo cuando **todos los hitos listados en [01-milestones.md](./01-milestones.md) estén `done`**.
- Merge via **PR merge** (preserva commits squashed por hito; facilita bisect).

### Conflictos con `robustness-feature-engineering`

Ambas ramas salen de `dev` y probablemente tocarán archivos distintos (RFE: `src/`, `notebooks/03`, `04`; viz: nuevos notebooks o celdas al final de `05`). Si hay conflicto:

1. Primero merge `robustness-feature-engineering` → `dev`.
2. Rebase `data-viz` sobre `dev` actualizado.
3. Resolver en sub-ramas de viz según corresponda.

## Housekeeping

- No eliminar sub-ramas `viz/hito-NN-*` tras merge — conservan la historia detallada del hito.
- `data-viz` se conserva hasta que haga merge a `dev`. Si no todos los hitos salen adelante, se hace merge de los que estén `done` y se cierra con decisión documentada en [01-milestones.md](./01-milestones.md).
