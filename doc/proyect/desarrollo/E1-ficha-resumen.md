# Ficha resumen — Entrega 1

**Proyecto**: Dinámica de precios residenciales en Dinamarca 1992–2024
**Curso**: Data Visualization — UPC, 2026-01

**Equipo**:

| Código     | Apellidos y nombres                 |
|------------|-------------------------------------|
| U202216562 | Vilchez Marín, Rody Sebastián       |
| U201520327 | Ballón Villar, Diego Eduardo        |
| U202218075 | Velásquez Borasino, Christian Aaron |

---

## Pregunta analítica

¿Cómo se han diferenciado los precios residenciales entre la región capital y las provincias danesas bajo distintos regímenes de tasas de interés e inflación, y qué tipologías de vivienda muestran mayor volatilidad y peores *drawdowns* durante las crisis financieras entre 1992 y 2024?

**Alcance**: análisis descriptivo y comparativo. Las asociaciones observadas son consistentes con mecanismos económicos conocidos, pero **no constituyen evidencia causal**; no se aplica diseño de identificación en E1.

## KPI rector

- **Headline**: precio real por m² (deflactado por inflación, base 2024).
- **Comparativo regional**: índice regional base 1992 Q1 = 100.
- *Drill-downs*: volumen transaccional, drawdown pico-valle, volatilidad rolling 4Q, elasticidad volumen→bonos con rezagos 0/1/2 Q.

## Usuario objetivo primario

**Inversor inmobiliario** con mandato sobre residencial danés. Requiere comparar riesgo/retorno entre regiones y tipologías bajo distintos regímenes macro y realizar *deep-dives* hasta nivel de código postal y año de construcción para decisiones de asignación de capital.

*Usuarios secundarios*: analista de riesgo crediticio bancario (H1) y consultor de políticas públicas (equidad regional).

## Dataset

- **Fuente**: Kaggle — Martin Frederiksen (~1.5 M registros, 1992–2024).
- **Link**: https://www.kaggle.com/datasets/martinfrederiksen/danish-residential-housing-prices-1992-2024/data
- **Granularidad**: transacción individual de compraventa residencial (fecha + dirección).
- **Dimensiones clave**: temporal (`date`, `quarter`), geográfica (`zip_code`, `region`), categórica (`house_type`, `sales_type`), macroeconómica (tasas e inflación trimestrales).

## Hipótesis iniciales

- **H1**: relación **negativa significativa** entre rendimiento de bonos hipotecarios y volumen transaccional, posiblemente con rezago de 1–2 trimestres. La magnitud se estimará descriptivamente (sin valor puntual adelantado).
- **H2**: Copenhague muestra respuesta más amortiguada de precios ante shocks macro que Jutlandia, consistente con restricciones de oferta.
- **H3**: Las casas de verano (`Summerhouse`) exhiben caídas más agudas y *drawdowns* mayores que las viviendas primarias durante recesiones.

## Valor potencial

- Cuantifica la sensibilidad del mercado residencial a shocks macro sobre un horizonte de 32 años.
- Permite identificar regiones y tipologías sobrevaloradas para apoyar decisiones de inversión.
- Integra overlay macro (tasas, inflación, bonos hipotecarios) con precios a nivel trimestral.
- Entrega un dashboard con drill-down hasta código postal y año de construcción, reutilizable por perfiles bancarios y de política pública.

## Entregables de esta entrega

- Informe completo: [`E1.md`](./E1.md)
- Ficha resumen: este documento
- Dataset: link en sección *Dataset*
