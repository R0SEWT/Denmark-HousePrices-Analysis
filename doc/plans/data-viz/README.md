# Data Viz — Paraguas de desarrollo analítico por hitos

Plan para desarrollar el ángulo visual/analítico del proyecto, estructurado en **hitos independientes**. Esta rama **no implementa visualizaciones**; define el contrato para que sub-ramas hijas (una por hito) ejecuten la implementación y hagan PR de vuelta.

> **Rama**: `data-viz` (sale de `dev`)
> **Rol**: paraguas — agrega PRs de sub-ramas `viz/hito-NN-<slug>`
> **Target de merge final**: `data-viz` → PR → `dev`

## Índice

| Documento | Propósito |
|---|---|
| [00-context.md](./00-context.md) | Objetivos analíticos: qué preguntas debe responder la viz |
| [01-milestones.md](./01-milestones.md) | Lista de hitos + slots para planificar cada uno |
| [02-branching-convention.md](./02-branching-convention.md) | Naming, estructura y merge policy de las sub-ramas |

## Flujo de trabajo

```
dev
 └── data-viz (paraguas)
      ├── viz/hito-01-geo               ← EDA geoespacial
      ├── viz/hito-02-series-temporales ← dinámica temporal regional
      └── viz/hito-03-model-comparison  ← dashboards comparativos modelos
```

1. Cada hito abre su propia sub-rama desde `data-viz`.
2. Antes de implementar código, añade su plan bajo `doc/plans/data-viz/hitos/NN-<slug>/` (mismo estilo docs-as-code que [robustness-feature-engineering](../robustness-feature-engineering/)).
3. PR de la sub-rama → `data-viz`. Review, merge.
4. Cuando todos los hitos terminen, `data-viz` → PR → `dev`.

## Estado actual

- [x] Scaffolding publicado
- [ ] Hito 1: EDA geoespacial (sin sub-rama aún)
- [ ] Hito 2: Series temporales regionales (sin sub-rama aún)
- [ ] Hito 3: Dashboards comparativos de modelos (sin sub-rama aún)

## Convenciones compartidas con `robustness-feature-engineering`

- Docs-as-code en `doc/plans/`.
- Frontmatter YAML en cada task doc con `id`, `title`, `branch`, `status`, `depends_on`, `touches`.
- Commits prefijados con ID de hito (ej. `[VIZ-01] ...`).
- Entorno: `source /shared/Code/hackathon-participants/.venv/bin/activate`.
