# Trabajo Final del Curso

Curso: `Data Visualization`  
Herramientas principales: `Tableau` y `Python` con notebooks

## 1. Propósito del trabajo final

El trabajo final del curso consiste en desarrollar un producto integral de visualización de datos orientado a la toma de decisiones. El proyecto debe mostrar que el equipo puede recorrer todo el flujo de trabajo visto en clase: comprender un problema, evaluar una fuente de datos, perfilarla, limpiarla, modelarla, analizarla, diseñar visualizaciones adecuadas, construir un dashboard funcional y defender técnicamente sus decisiones.

El objetivo no es solo presentar gráficos atractivos. El objetivo es construir una solución analítica sólida, reproducible y bien argumentada.

## 2. Modalidad de trabajo

- Equipos de `2` a `3` estudiantes.
- Si el docente lo aprueba, puede adaptarse a modalidad individual.
- El equipo debe trabajar un solo tema durante todo el ciclo.
- Cada entrega parcial debe reutilizar y mejorar la entrega anterior.

## 3. Producto final esperado

Al final del curso, cada equipo deberá entregar:

- Un `dashboard final` en `Tableau`.
- Una `historia visual` breve o secuencia narrativa equivalente.
- Un conjunto de `notebooks` en `Python` con el pipeline del proyecto.
- Los archivos de datos preparados para Tableau.
- Una bitácora técnica con las decisiones de preparación, validación y modelado.
- Una presentación final para la defensa del proyecto.

## 4. Qué debe resolver el proyecto

El proyecto debe responder una pregunta analítica real y relevante. Esa pregunta debe estar vinculada con una decisión, necesidad de monitoreo, evaluación de desempeño, identificación de patrones o comprensión de un fenómeno.

El equipo debe poder responder con claridad:

- ¿Cuál es el problema que se quiere analizar?
- ¿Quién usará el dashboard?
- ¿Qué decisión o acción podría apoyarse con este análisis?
- ¿Por qué el dataset elegido permite responder esa pregunta?

## 5. Requisitos del dataset

El dataset debe cumplir, idealmente, con las siguientes condiciones:

- Tener al menos `2,000` registros.
- Tener al menos `10` variables útiles.
- Incluir al menos `1` variable temporal.
- Incluir al menos `1` variable categórica relevante.
- Incluir al menos `1` variable que permita segmentar.
- Incluir al menos `4` variables numéricas.
- De preferencia, incluir una dimensión geográfica.
- Si se quiere aplicar `PCA` o `t-SNE`, contar con suficientes variables numéricas o una representación adecuada.

Además:

- Debe ser una fuente real, pública, institucional o verificable.
- Debe tener trazabilidad: origen, periodo, cobertura y limitaciones.
- No debe ser un dataset trivial.
- No debe ser un dataset tan limpio que impida trabajar perfilado y limpieza.

## 6. Cobertura mínima obligatoria del curso

El trabajo final debe evidenciar todos los siguientes componentes:

1. Definición del problema y del usuario objetivo.
2. Perfilado del dato y comprensión de la granularidad.
3. Limpieza y preparación.
4. Modelado analítico o estructuración de la fuente.
5. Análisis exploratorio.
6. Selección y justificación de gráficos.
7. Segmentación e interpretación.
8. Construcción de métricas o cálculos analíticos.
9. Storytelling técnico.
10. Revisión de accesibilidad y diseño visual.
11. Visualización longitudinal.
12. Visualización transversal o comparativa.
13. Componente avanzado.
14. Construcción del dashboard final.
15. Defensa metodológica y visual.

## 7. Configuración mínima del producto final

Como mínimo, el proyecto final debe contener:

- `1` problema analítico claramente formulado.
- `1` usuario objetivo principal.
- `1` dataset principal validado.
- `1` pipeline reproducible en `Python`.
- `1` workbook en `Tableau` con al menos `8` hojas analíticas útiles.
- `1` dashboard final con al menos `3` bloques funcionales:
  - contexto y KPIs
  - exploración comparativa o segmentada
  - módulo temporal, comparativo o avanzado
- `1` historia visual de `3` a `5` pantallas o una secuencia equivalente.
- `1` vista temporal sólida.
- `1` vista transversal sólida.
- `1` componente avanzado integrado o anexado con justificación metodológica.
- `1` documento breve de QA con validación técnica, limitaciones y decisiones de diseño.

## 8. Entregas parciales

El trabajo final se desarrollará de manera progresiva mediante entregas parciales.

### Entrega 1. Propuesta del proyecto

Semana sugerida: `2`

Debe incluir:

- tema del proyecto
- pregunta analítica principal
- usuario objetivo
- fuente de datos propuesta
- hipótesis iniciales
- valor potencial del proyecto

Entregables:

- documento breve en `PDF`
- enlace o archivo del dataset
- ficha resumen del proyecto

Criterios mínimos:

- la pregunta analítica no es trivial
- el usuario objetivo está claramente definido
- el dataset permite temporalidad, segmentación y comparación
- se explican riesgos iniciales de calidad o cobertura

### Entrega 2. Perfilado, diccionario y limpieza inicial

Semana sugerida: `4`

Debe incluir:

- perfilado del dataset
- unidad de análisis
- granularidad
- diccionario de datos
- principales problemas de calidad
- reglas de limpieza
- bitácora inicial

Entregables:

- notebook de perfilado y limpieza
- tabla de perfilado
- dataset limpio preliminar
- bitácora de transformaciones

Criterios mínimos:

- se reportan nulos, cardinalidad, duplicados y campos críticos
- se justifican las decisiones de limpieza
- el dataset limpio puede conectarse a Tableau sin ambigüedad
- se diferencia claramente la fuente original de la fuente transformada

### Entrega 3. Exploración, chart selection e insights iniciales

Semana sugerida: `6`

Debe incluir:

- primeras visualizaciones exploratorias
- comparación, distribución, relación y tendencia
- criterios de selección de gráficos
- primeros insights

Entregables:

- notebook exploratorio
- workbook preliminar en Tableau
- documento corto con `3` a `5` insights

Criterios mínimos:

- se muestran distintas familias de gráficos
- cada gráfico tiene una justificación técnica breve
- al menos `2` opciones visuales fueron descartadas y explicadas
- los insights están redactados con lenguaje analítico

### Entrega 4. Modelado, métricas y dashboard alpha

Semana sugerida: `8`

Debe incluir:

- estructura analítica o relacional
- validación de joins o relationships
- métricas derivadas
- segmentación
- parámetros, filtros o lógica analítica
- primera versión funcional del dashboard

Entregables:

- notebook de modelado y cálculos
- fuentes analíticas para Tableau
- dashboard alpha

Criterios mínimos:

- la estructura relacional está validada
- las métricas responden a la pregunta del proyecto
- el dashboard alpha ya tiene flujo de lectura
- existe al menos una vista segmentada útil

### Entrega 5. Storytelling, accesibilidad y módulo temporal/comparativo

Semana sugerida: `10`

Debe incluir:

- rediseño narrativo del dashboard
- mejora de accesibilidad
- vista longitudinal sólida
- vista transversal sólida

Entregables:

- dashboard revisado
- historia visual breve
- checklist de accesibilidad
- reflexión metodológica corta

Criterios mínimos:

- mejora visible respecto a la versión alpha
- títulos, color, contraste y jerarquía están mejor resueltos
- la vista temporal y la vista transversal son defendibles
- la historia visual comunica un mensaje, no solo muestra gráficos

### Entrega 6. Componente avanzado y dashboard beta

Semana sugerida: `12`

Debe incluir:

- aplicación de `PCA`, `t-SNE` o técnica equivalente aprobada
- explicación metodológica
- integración del resultado al análisis
- versión beta del dashboard

Entregables:

- notebook avanzado
- exportables para Tableau
- vista avanzada integrada o anexa
- dashboard beta

Criterios mínimos:

- la técnica está correctamente explicada
- se documentan variables, parámetros y limitaciones
- el resultado aporta valor analítico al caso
- el dashboard beta ya está listo para revisión final

### Entrega 7. Entrega final y defensa

Semana sugerida: `14`

Debe incluir:

- dashboard final
- historia visual o secuencia narrativa
- QA técnico
- defensa integral del pipeline

Entregables:

- dashboard final en Tableau
- carpeta final de notebooks
- base final preparada
- presentación de defensa
- resumen ejecutivo

Criterios mínimos:

- el dashboard responde claramente la pregunta principal
- el pipeline técnico es reproducible
- el equipo demuestra cobertura de todo lo visto en clase
- la defensa muestra dominio de supuestos, límites y decisiones

## 9. Cronograma resumido

| Semana | Tema del curso             | Hito del proyecto        |
| ------ | -------------------------- | ------------------------ |
| 1      | Introducción y ecosistema  | Exploración de temas     |
| 2      | Perfilado y granularidad   | Entrega `1`              |
| 3      | Limpieza y preparación     | Avance técnico           |
| 4      | Modelado y fuentes         | Entrega `2`              |
| 5      | Visualización exploratoria | Prototipos               |
| 6      | Segmentación e insights    | Entrega `3`              |
| 7      | Cálculos analíticos        | Construcción de métricas |
| 8      | Storytelling técnico       | Entrega `4`              |
| 9      | Accesibilidad y diseño     | Revisión visual          |
| 10     | Series temporales          | Entrega `5`              |
| 11     | Comparación transversal    | Integración comparativa  |
| 12     | `PCA` y `t-SNE`            | Entrega `6`              |
| 13     | Dashboard engineering y QA | Cierre técnico           |
| 14     | Capstone y defensa         | Entrega `7`              |

## 10. Criterios globales de evaluación

La evaluación del trabajo final considerará, como mínimo, los siguientes aspectos:

- definición del problema y claridad del objetivo
- calidad, trazabilidad y perfilado del dato
- limpieza y preparación reproducible
- modelado y consistencia técnica
- rigor analítico e insights
- calidad visual y accesibilidad
- storytelling y comunicación
- componente avanzado
- defensa final

## 11. Formato de entrega sugerido

Se recomienda organizar la entrega final de esta forma:

```text
proyecto-final/
  data/
  notebooks/
  outputs/
  tableau/
  docs/
  README.md
```

Contenido esperado:

- `data/`: fuentes originales y fuentes limpias
- `notebooks/`: notebooks de trabajo
- `outputs/`: exportables para Tableau
- `tableau/`: workbook o enlace publicado
- `docs/`: diccionario, bitácora, QA, presentación y resumen ejecutivo
- `README.md`: explicación general del proyecto

## 12. Relación con los laboratorios del curso

Cada entrega parcial debe apoyarse en los laboratorios y notebooks semanales del curso. La idea es que el trabajo final no se construya al final del semestre como un esfuerzo aislado, sino como una acumulación ordenada de evidencia técnica y analítica.

Se recomienda usar como base:

- perfilado y granularidad
- limpieza y preparación
- modelado de fuentes
- chart selection
- segmentación e insights
- cálculos analíticos
- storytelling y anotación
- accesibilidad y diseño
- análisis temporal
- comparación transversal
- `PCA` o `t-SNE`
- dashboard engineering
- QA final

## 13. Recomendaciones importantes

- Elijan una pregunta concreta, no un tema demasiado amplio.
- No construyan el dashboard antes de entender la granularidad del dato.
- No usen gráficos por variedad visual; úsenlos porque responden mejor a una pregunta.
- No oculten problemas de calidad; documentarlos también es parte del trabajo.
- No usen el componente avanzado como adorno técnico; debe aportar al análisis.
- Cuiden la legibilidad, el contraste y el orden visual del dashboard.
- Ensayen la defensa con foco en decisiones, no solo en resultados.

## 14. Qué se espera de una buena entrega final

Una buena entrega final:

- responde una pregunta real
- usa datos con trazabilidad
- documenta su pipeline
- construye insights defendibles
- comunica con claridad
- y puede sostener técnicamente lo que afirma
