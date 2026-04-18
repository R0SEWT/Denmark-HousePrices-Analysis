# 00 — Contexto analítico

## ¿Qué preguntas debe responder la viz?

El proyecto es un análisis académico sobre precios residenciales daneses 1992–2024. La viz complementa el modelado (ver [robustness-feature-engineering](../robustness-feature-engineering/)) para:

1. **Narrar el dataset**: mostrar a un lector qué es, cómo se distribuye, qué dinámica temporal tiene.
2. **Justificar decisiones de modelado**: ¿por qué log_price?, ¿por qué split temporal?, ¿por qué ventana rolling?
3. **Contextualizar resultados**: ¿dónde acierta el modelo?, ¿dónde falla?, ¿qué regiones/períodos son más difíciles?
4. **Comunicar a un público académico**: las visualizaciones finales irán en `doc/memoria/` como parte del informe.

## Preguntas concretas (shortlist)

### Geoespacial
- Distribución de precios por región (Zealand vs Jutland, urbano vs rural).
- ¿Dónde ocurren las transacciones? (densidad)
- ¿Qué regiones tienen mayor volatilidad de precio?

### Temporal
- Serie histórica de mediana de precio por región.
- Impacto observable de eventos macro (crisis 2008, covid 2020–2021).
- Estacionalidad intra-año en precio y volumen de transacciones.

### Modelado
- Distribución de errores del modelo final por región / año.
- Importancia de features (SHAP top-N).
- Comparación predicción vs real en holdout 2023–2024.

## Restricciones

- **No reprocessar** el dataset crudo — consumir el parquet generado por notebook 03.
- **Respetar español** en toda la salida (títulos, ejes, leyendas).
- **No reintroducir leaks**: si una viz necesita `purchase_price` crudo, cargarlo solo para esa celda, no propagarlo al pipeline.
- **Outputs**: PNG/SVG a `results/charts/`; tablas a `results/tablas/`. Ambos dirs están gitignoreados, así que los charts finales se referencian desde `doc/memoria/` vía copia explícita.

## Relación con otros planes

- Depende de que `robustness-feature-engineering` haya publicado un parquet saneado en `data/processed/processed_data.parquet` (ver [RFE-07](../robustness-feature-engineering/tasks/07-notebook-03.md)).
- Hito 3 (comparación de modelos) depende de que el modelo final esté guardado en `models/` ([RFE-08](../robustness-feature-engineering/tasks/08-notebook-04.md)).
