# Documentación del Módulo de Análisis Refactorizado

## Estructura del Proyecto

```
src/
├── __init__.py                 # Inicialización del paquete
├── utils.py                    # Módulo principal refactorizado
├── utils_backup.py            # Backup del archivo original
└── analysis/                  # Módulos especializados
    ├── __init__.py
    ├── data_quality.py        # Análisis de calidad de datos
    ├── univariate_analysis.py # Análisis univariado básico
    ├── enhanced_analysis.py   # Análisis univariado avanzado
    ├── visualization.py       # Funciones de visualización
    └── summary_analysis.py    # Análisis y resúmenes consolidados
```

## Módulos Especializados

### 1. `data_quality.py`
**Propósito**: Análisis de calidad de datos
**Funciones principales**:
- `get_df_null_resume_and_percentages()`: Resumen de valores nulos
- `get_duplicate_percentage()`: Porcentaje de duplicados
- `plot_null_heatmap()`: Mapa de calor de valores nulos
- `get_column_types()`: Información de tipos de columnas
- `analyze_data_quality()`: Análisis completo de calidad

### 2. `univariate_analysis.py`
**Propósito**: Análisis univariado básico
**Funciones principales**:
- `analyze_numeric_series()`: Análisis de series numéricas
- `plot_numeric_distribution()`: Visualización de distribuciones numéricas
- `describe_numeric()`: Descripción de variables numéricas
- `plot_discrete_distribution()`: Visualización de variables discretas
- `describe_discrete()`: Descripción de variables discretas
- `analizar_variable_categorica()`: Análisis de variables categóricas
- `run_univariate_analysis()`: Función principal de análisis univariado

### 3. `enhanced_analysis.py`
**Propósito**: Análisis univariado avanzado con insights de negocio
**Funciones principales**:
- `enhanced_univariate_analysis()`: Análisis detallado con insights
- Funciones auxiliares para estadísticas avanzadas y visualizaciones

### 4. `visualization.py`
**Propósito**: Funciones especializadas de visualización
**Funciones principales**:
- `plot_target_distribution()`: Distribución de variable objetivo
- `create_correlation_heatmap()`: Mapa de correlaciones
- `create_distribution_comparison()`: Comparación de distribuciones
- `create_boxplot_comparison()`: Comparación de boxplots
- `create_categorical_summary_plot()`: Resumen de categóricas
- `create_missing_data_visualization()`: Visualización de datos faltantes
- `create_outlier_visualization()`: Visualización de outliers
- `create_advanced_univariate_dashboard()`: Dashboard avanzado

### 5. `summary_analysis.py`
**Propósito**: Análisis y resúmenes consolidados
**Funciones principales**:
- `create_univariate_summary()`: Resumen univariado consolidado
- `create_data_quality_report()`: Reporte de calidad de datos
- `create_correlation_analysis()`: Análisis de correlaciones

## Funciones Nuevas en utils.py

### `run_complete_analysis(df, target_column=None)`
Ejecuta un análisis completo del DataFrame incluyendo:
- Análisis de calidad de datos
- Análisis univariado
- Análisis de correlaciones
- Dashboard avanzado
- Resumen consolidado

### `quick_analysis(df, columns=None, max_cols=5)`
Análisis rápido de columnas específicas o selección automática.

### `get_preprocessing_recommendations(df)`
Obtiene recomendaciones de preprocesamiento basadas en el análisis:
- Tratamiento de datos faltantes
- Detección de outliers
- Necesidad de transformaciones
- Recomendaciones de encoding
- Sugerencias de escalado

### `generate_html_report(df, output_path="analysis_report.html")`
Genera un reporte HTML con el análisis completo (en desarrollo).

## Ventajas de la Refactorización

1. **Modularidad**: Cada módulo tiene una responsabilidad específica
2. **Mantenibilidad**: Código más fácil de mantener y debugear
3. **Reutilización**: Funciones pueden ser importadas individualmente
4. **Escalabilidad**: Fácil agregar nuevas funcionalidades
5. **Legibilidad**: Código más organizado y documentado
6. **Testing**: Cada módulo puede ser testeado independientemente

## Uso Recomendado

```python
# Importar el módulo completo
import sys
sys.path.append('./src')
import utils

# Análisis completo
results = utils.run_complete_analysis(df, target_column='purchase_price')

# Análisis rápido
quick_results = utils.quick_analysis(df, columns=['price', 'sqm', 'region'])

# Recomendaciones
recommendations = utils.get_preprocessing_recommendations(df)

# Funciones específicas
utils.enhanced_univariate_analysis(df, 'purchase_price', 'numeric')
```

## Compatibilidad

El nuevo módulo mantiene compatibilidad con el código existente. Todas las funciones originales están disponibles a través de los imports modulares.

## Próximos Pasos

1. **Testing**: Implementar tests unitarios para cada módulo
2. **Documentación**: Agregar docstrings detallados
3. **Reportes HTML**: Completar la funcionalidad de reportes
4. **Análisis Multivariado**: Agregar módulo de análisis multivariado
5. **Optimización**: Mejorar performance para datasets grandes
