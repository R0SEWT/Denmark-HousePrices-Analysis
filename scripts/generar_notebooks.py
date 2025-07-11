from pathlib import Path
import nbformat as nbf

# Carpeta de notebooks
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

# Bloque de código común (setup de paths)
path_setup_code = '''\
from pathlib import Path
import os
import sys

current_path = Path.cwd()
if current_path.name == "notebooks":
    project_root = current_path.parent
else:
    project_root = current_path

os.chdir(project_root)
print(f"Current working directory: {Path.cwd()}")

sys.path.append(str(project_root / "src"))
'''

# Estructura de notebooks
notebooks = {
    "02_analisis_descriptivo.ipynb": [
        "# 02 - Análisis Descriptivo",
        "## Objetivo\nAnálisis de KPIs, evolución temporal y diferencias regionales.",
        "## 1. KPIs por región",
        "## 2. Evolución temporal de precios",
        "## 3. Diferencias por tipo de propiedad",
        "## 4. Visualizaciones relevantes",
        "## 5. Conclusiones parciales"
    ],
    "03_feature_engineering.ipynb": [
        "# 03 - Feature Engineering",
        "## Objetivo\nConstrucción y transformación de variables predictivas.",
        "## 1. Transformación de tipos y columnas derivadas",
        "## 2. Codificación de variables categóricas",
        "## 3. Normalización y escalado",
        "## 4. Dataset final para modelado",
        "## 5. Guardado del dataset limpio"
    ],
    "04_modelado_supervisado.ipynb": [
        "# 04 - Modelado Supervisado",
        "## Objetivo\nEntrenamiento y evaluación de modelos para predecir precios.",
        "## 1. División de datos (train/test)",
        "## 2. Modelo base: Regresión Lineal (GLM)",
        "## 3. Modelo complejo: Gradient Boosting (GBM)",
        "## 4. Comparación de métricas",
        "## 5. Interpretación con SHAP y/o LIME"
    ],
    "05_resultados_finales.ipynb": [
        "# 05 - Resultados Finales",
        "## Objetivo\nResumen de métricas, hallazgos y visualizaciones clave.",
        "## 1. Tabla comparativa de modelos",
        "## 2. Gráficos de error (hist, scatter, etc.)",
        "## 3. Principales insights del modelo",
        "## 4. Recomendaciones y próximos pasos"
    ]
}

# Generar notebooks
for name, markdown_cells in notebooks.items():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_code_cell(path_setup_code))
    nb.cells += [nbf.v4.new_markdown_cell(cell) for cell in markdown_cells]
    path = notebooks_dir / name
    with path.open("w") as f:
        nbf.write(nb, f)

print(f"✅ Notebooks generados en {notebooks_dir.resolve()}")
