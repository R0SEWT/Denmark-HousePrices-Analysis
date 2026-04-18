# 01 — Hitos (milestones)

Cada hito es una entrega atómica ejecutada en su propia sub-rama. El plan detallado de cada uno vive en `doc/plans/data-viz/hitos/NN-<slug>/` (estructura a crear **dentro** de la sub-rama del hito, no aquí).

## Hito 1 — EDA visual geoespacial

| Atributo | Valor |
|---|---|
| ID | VIZ-01 |
| Sub-rama | `viz/hito-01-geo` (salir de `data-viz`) |
| Estado | planificación pendiente |
| Depende de | — (puede iniciar en paralelo con RFE) |

**Entregables esperados**:
- Mapa de Dinamarca con densidad de transacciones por región/zip.
- Choropleth de mediana de precio por región (con slider o facetas por década).
- Boxplot de precio por `house_type` × `region_cluster`.
- Notebook o sección en notebook 02 que genere los charts.
- Charts exportados a `results/charts/geo/`.

**Restricciones**:
- Si hay columnas lat/lon aproximadas (zip_code → centroid), usarlas; si no, mapear región a polígonos vía un shapefile externo.
- Paleta accesible (evitar rojo-verde; usar `viridis` o `cividis`).

---

## Hito 2 — Series temporales regionales

| Atributo | Valor |
|---|---|
| ID | VIZ-02 |
| Sub-rama | `viz/hito-02-series-temporales` |
| Estado | planificación pendiente |
| Depende de | RFE-07 (parquet saneado con `year` y agregados rolling) |

**Entregables esperados**:
- Línea de mediana de precio por región × año.
- Overlay de eventos macro (crisis 2008, covid).
- Pequeños múltiples (small multiples) por `house_type`.
- Comparación de `rolling_regional_mean` (ventana k=3) vs curva real por región.
- Análisis de cambios de régimen: detectar segmentos de tendencia distinta.

**Restricciones**:
- Convertir `log_price` → DKK vía `np.expm1` antes de mostrar.
- Eje Y en log o lineal según la narrativa; documentar la decisión en el notebook.

---

## Hito 3 — Dashboards comparativos de modelos

| Atributo | Valor |
|---|---|
| ID | VIZ-03 |
| Sub-rama | `viz/hito-03-model-comparison` |
| Estado | planificación pendiente |
| Depende de | RFE-08 (modelo final guardado + métricas walk-forward) |

**Entregables esperados**:
- Scatter predicción vs real en holdout 2023–2024, coloreado por región.
- Histograma de residuales (log_price y DKK).
- Tabla de métricas por año del holdout (2023, 2024) y por región top-N.
- Plot SHAP top-15 con anotaciones.
- Comparación de RMSE entre: modelo legacy (single split, con leak), walk-forward CV mean, holdout final.

**Restricciones**:
- NO reentrenar modelos en este hito; consumir lo que ya está guardado.
- Si existe más de un modelo candidato (ej. con y sin `is_premium_causal`), comparar ambos.

---

## Plantilla para proponer un hito nuevo

Copiar al final de esta lista:

```markdown
## Hito N — <título corto>

| Atributo | Valor |
|---|---|
| ID | VIZ-NN |
| Sub-rama | `viz/hito-NN-<slug>` |
| Estado | planificación pendiente |
| Depende de | <lista de IDs> |

**Entregables esperados**: ...
**Restricciones**: ...
```
