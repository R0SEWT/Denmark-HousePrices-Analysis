
## Estrcutra de proyecto

``` css
TFBigData/
├── data/                      ← Datos brutos y procesados
│   ├── DKHousingPrices.parquet
│   ├── DKHousingPricesSample100k.csv
│   ├── processed/             ← Datos limpios / con features
│   └── external/              ← Datos de otra fuente (geo, inflación, etc.)

├── notebooks/                 ← Análisis completo en .ipynb
│   ├── 00_extraccion_de_datos.ipynb
│   ├── 01_exploracion_eda.ipynb
│   ├── 02_analisis_descriptivo.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelado_supervisado.ipynb
│   ├── 05_resultados_finales.ipynb

├── src/                       ← Funciones reutilizables
│   ├── analysis/              ← Análisis de datos (EDA, visuales, etc.)
│   ├── utils.py               ← Funciones auxiliares generales
│   ├── config.py              ← Configuración global (rutas, columnas, etc.)

├── models/                    ← Modelos entrenados
│   └──    

├── metrics/                   ← Métricas csv, json o .db (optuna)
│   └── 

├── results/                   ← Visualizaciones y tablas para el informe
│   ├── charts/
│   └── tablas/

├── utils/                     ← Recursos complementarios y documentacion automatica del proyecto
│   └── doc_src/               ← Diagramas y esquemas del flujo

├── memoria/                   ← Informe final en notebook, PDF o MD
│   ├── informe_final.pdf
│   └── informe_final.md

├── README.md                  ← Descripción del proyecto y cómo ejecutarlo
├── environment.yml            ← Dependencias para reproducir el entorno
└── run_all.sh                 ← Script para correr todos los notebooks (opcional)

```