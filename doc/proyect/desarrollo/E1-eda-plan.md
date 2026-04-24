# Plan — EDA de respaldo para Entrega 1

| Atributo | Valor |
|---|---|
| ID | E1-EDA |
| Entregable padre | Entrega 1 — Propuesta del proyecto ([`E1.md`](./E1.md)) |
| Tipo | Evidencia empírica (figuras + tablas) |
| Rama | `data-viz` |
| Estado | planificado |
| Salidas | `notebooks/eda_e1/01_eda_propuesta.ipynb` + artefactos en `results/charts/e1/` y `results/tablas/e1/` (gitignoreados) |

## 1. Motivación

El informe [`E1.md`](./E1.md) afirma claims cuantitativos (shape del dataset, % nulos, KPIs, divergencia regional, efecto de shocks, volatilidad por tipología) sin evidencia empírica embebida. Los notebooks existentes (`notebooks/01_exploracion_eda.ipynb`, `notebooks/02_analisis_descriptivo.ipynb`) cubren una parte, pero dejan gaps concretos que deben respaldarse antes de la defensa de E1.

### Cobertura existente (no re-hacer)

- Shape y rango temporal del dataset.
- % de nulos por columna.
- % de `sales_type = '-'`.
- Análisis regional de precios nominales.
- Distribuciones por `house_type` (básicas).

### Gaps cubiertos por este EDA

1. **Precio real por m² deflactado** (KPI headline, §5.1 de E1).
2. **Índice regional base 1992 Q1 = 100** (KPI comparativo, §5.1 de E1).
3. **Granularidad trimestral con shock markers** (1995 / 2008 Q3 / 2020 Q2 / 2022 Q1), §5.2 de E1.
4. **Drawdowns pico-valle por `house_type`** (evidencia H3, §6 de E1).
5. **Umbral cuantitativo de Bornholm** (<200 tx/año, §8 riesgos de E1).
6. **Diferencial de completitud pre/post-1995** (sesgo en registros preliminares, §8 de E1).

## 2. Decisiones fijadas

| # | Decisión | Racional |
|---|---|---|
| D1 | Ubicación del notebook: `notebooks/eda_e1/01_eda_propuesta.ipynb` | Subcarpeta nueva bajo `notebooks/` — separada del pipeline 00–05 pero dentro del directorio convencional. Escala a futuros EDA por entrega |
| D2 | Alcance: gaps listados arriba + re-verificación rápida de claims | No duplicar 01/02 pero sí confirmar numéricamente lo que el informe cita |
| D3 | Helpers (deflactor, índice, drawdown) viven **inline en el notebook** | Menor blast radius. Promoción a `src/analysis/kpis.py` solo si hitos posteriores los reusan |
| D4 | Deflactor: CPI interno compuesto desde `dk_ann_infl_rate%` + nota metodológica | Evita dependencia externa en E1. La versión oficial (CPI de Statistics Denmark) queda para E2 |
| D5 | Paleta `viridis`/`cividis`, etiquetas en español | Convención data-viz documentada en CLAUDE.md |

## 3. Estructura del notebook

### Celda 0 — Setup explícito

`notebooks/setup.py` chequea `cwd.name == "notebooks"`; desde `notebooks/eda_e1/` ese chequeo falla. Para no tocar `setup.py` (afectaría a los notebooks 00–05), el arranque es explícito:

```python
import sys, os
from pathlib import Path

project_root = Path.cwd()
while not (project_root / "src").exists() and project_root != project_root.parent:
    project_root = project_root.parent
os.chdir(project_root)
sys.path.append(str(project_root / "src"))

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
```

### Celda 1 — Carga del raw parquet

Respetar `ISDISTRIBUTED` de `src/config.py`: usar `DATA_FILE` o `DISTRIBUTED_DATA_FILE` según el flag. Cargar con `pyarrow`/`pandas`. Sin feature engineering — el EDA opera sobre `data/raw/DKHousingPrices.parquet`, no sobre `processed_data.parquet`.

### Secciones analíticas

**§1. Shape y cobertura temporal** — conteo de filas, `date.min/max`, % nulos por columna (reusar `src.analysis.data_quality.get_df_null_resume_and_percentages`, expuesto en `src/utils.py`). Output: tabla inline + `results/tablas/e1/shape_nulls.csv`.

**§2. `sales_type='-'` y completitud pre-1995** — porcentaje exacto de `sales_type='-'`; matriz de missing-rate en `year_build` y `sqm` por cohorte anual 1992–2024 para sustentar el corte 1995. Output: heatmap `results/charts/e1/completitud_pre_1995.png`.

**§3. Cobertura regional y umbral Bornholm** — transacciones por año × región; validar claim "Bornholm <200 tx/año" y umbral mínimo de 50 tx/año usado en §8 del informe. Output: `results/charts/e1/cobertura_regional.png` + `results/tablas/e1/cobertura_regional.csv`.

**§4. Deflactor CPI interno** — helper inline `cpi_desde_inflacion_anual(df, rate_col, base_year=2024)`:

- Toma una observación anual (primer trimestre disponible).
- Compone `cpi_t = cpi_{t-1} * (1 + rate_t)`, reescala a `cpi[base_year] = 100`.
- Devuelve `pd.Series` indexada por año.

Incluir celda markdown con **nota metodológica**: la serie `dk_ann_infl_rate%` es tasa anual reportada trimestralmente; el CPI reconstruido es aproximación razonable; reemplazable por CPI oficial en E2. Output: `results/tablas/e1/cpi_interno.csv`.

**§5. KPI headline — precio real por m²** — helper inline `precio_real_m2(df, cpi_series, base_year=2024)`:

- `precio_real = purchase_price * (100 / cpi[año])` → DKK 2024.
- `precio_real_m2 = precio_real / sqm`.

Distribución por año (boxplot facetado) + distribución por `house_type`. Outputs: `results/charts/e1/precio_real_m2_distribucion.png`, `results/charts/e1/precio_real_m2_por_tipo.png`.

**§6. KPI comparativo — índice regional base 1992 Q1 = 100** — helper inline `indice_regional(df, precio_col="precio_real_m2", base_quarter="1992Q1")`:

- Agrega media trimestral por región.
- Reescala cada región dividiendo por el valor en `base_quarter` y multiplicando por 100.

Plot: líneas por región, suavizado rolling 4Q, shock markers verticales anotados (1995 / 2008 Q3 / 2020 Q2 / 2022 Q1), paleta `viridis`. Output: `results/charts/e1/indice_regional.png` + `results/tablas/e1/indice_regional.csv`.

**§7. Drawdowns pico-valle por `house_type`** (soporte H3) — helper inline `drawdown_por_tipo(df_index_trimestral)`:

- Para cada tipo, serie trimestral del índice base 100, luego `drawdown_t = indice_t / cummax(indice) - 1`.
- Reportar `max_drawdown` y ventana temporal donde ocurre.

Plot: curvas de drawdown por tipo + tabla resumen. Outputs: `results/charts/e1/drawdowns_por_tipo.png`, `results/tablas/e1/drawdowns_resumen.csv`.

**§8. Crosswalk claim → evidencia** (celda final markdown, obligatoria):

| Claim E1 | Sección informe | Sección notebook | Archivo generado |
|---|---|---|---|
| ~1.5M filas, 1992–2024 | §4.3 | §1 | `results/tablas/e1/shape_nulls.csv` |
| Nulos <0.1% en críticas | §4.3 / §8 | §1 | ídem |
| `sales_type='-'` ~0.5% | §8 | §2 | — |
| Bornholm <200 tx/año | §8 | §3 | `results/tablas/e1/cobertura_regional.csv` |
| KPI precio real m² computable | §5.1 | §4 + §5 | `results/charts/e1/precio_real_m2_*.png` |
| Índice regional 1992 Q1=100 | §5.1 | §6 | `results/charts/e1/indice_regional.png` |
| Divergencia Copenhague vs resto | §2.2 / §5 | §6 | ídem |
| Shocks visibles 2008/2020/2022 | §5.2 | §6 | ídem |
| H3: drawdowns diferenciales por tipo | §6 (hipótesis) | §7 | `results/charts/e1/drawdowns_por_tipo.png` |
| Sin lat/lon nativas | §4.4 / §8 | §1 | — |

## 4. Fuera de alcance

- **H1 con scatter volumen↔bonos con rezagos** — queda para componente avanzado (Entrega 6) o un hito posterior.
- **Feature engineering, limpieza, modelado** — pipeline de notebooks 03/04/05.
- **Data-leak fix** — vive en rama separada.
- **Reemplazo del CPI interno por CPI oficial** — se evalúa en E2.

## 5. Dependencias

- Raw parquet disponible en `data/raw/DKHousingPrices.parquet` o `DISTRIBUTED_DIR/DKHousingPrices.parquet` (según `ISDISTRIBUTED`).
- Entorno con `pandas`, `matplotlib`, `seaborn`, `pyarrow`. Per CLAUDE.md: `source /shared/Code/hackathon-participants/.venv/bin/activate`.
- Utilidades existentes en `src/utils.py` (facade sobre `src/analysis/*`).

## 6. Archivos tocados

| Archivo | Acción |
|---|---|
| `notebooks/eda_e1/01_eda_propuesta.ipynb` | Create |
| `doc/proyect/desarrollo/E1-eda-plan.md` | Create (este documento) |
| `results/charts/e1/*.png` | Generados al ejecutar (gitignored) |
| `results/tablas/e1/*.csv` | Generados al ejecutar (gitignored) |

No se modifican: `src/`, `notebooks/setup.py`, notebooks 00–05, ni los otros documentos de `doc/`.

## 7. Verificación

1. Correr el notebook end-to-end desde un kernel limpio; 0 excepciones.
2. Confirmar aparición de los 4 `.csv` y 5 `.png` bajo `results/` tras ejecutar.
3. Validaciones numéricas mínimas:
   - Shape ≈ 1.5 M filas y rango temporal 1992–2024.
   - Claim "Bornholm <200/año" se cumple **o** se corrige en [`E1.md`](./E1.md) §8.
   - `cpi[2024] = 100`; valor en 1992 significativamente menor (orden de 50–70).
   - Índice regional 1992 Q1=100 por construcción; Copenhague muy por encima de Bornholm/Jutland al 2024.
   - `max_drawdown` de `Summerhouse` más profundo que el de `Villa`/`Apartment` (consistente con H3; si no, flaggear en [`E1.md`](./E1.md) §6).
4. Si una claim falla la verificación, **prioridad a actualizar [`E1.md`](./E1.md)** sobre preservar el texto original.
5. Ejecución NO requiere `data/processed/processed_data.parquet` (output del pipeline 03, fuera de alcance).

## 8. Promoción posterior

Si en hitos `viz/hito-NN-*` los helpers `cpi_desde_inflacion_anual`, `precio_real_m2`, `indice_regional` o `drawdown_por_tipo` se vuelven a necesitar, promoverlos entonces a `src/analysis/kpis.py` y exponerlos via `src/utils.py`. No antes — evitar abstracción prematura.
