"""
Módulo de utilidades refactorizado y modularizado
Importa funciones de módulos especializados para análisis de datos
"""

# Importar módulos de análisis
from .analysis.data_quality import (
    get_df_null_resume_and_percentages,
    get_duplicate_percentage,
    plot_null_heatmap,
    get_column_types,
    analyze_data_quality
)

from .analysis.univariate_analysis import (
    analyze_numeric_series,
    plot_numeric_distribution,
    describe_numeric,
    plot_discrete_distribution,
    describe_discrete,
    obtener_top_y_otros,
    generar_wordcloud,
    analizar_high_cardinality,
    analizar_variable_categorica,
    plot_categorical_distributions,
    run_univariate_analysis
)

from .analysis.enhanced_analysis import (
    enhanced_univariate_analysis
)

from .analysis.visualization import (
    plot_target_distribution,
    create_correlation_heatmap,
    create_distribution_comparison,
    create_boxplot_comparison,
    create_categorical_summary_plot,
    create_missing_data_visualization,
    create_outlier_visualization,
    create_advanced_univariate_dashboard
)

from .analysis.summary_analysis import (
    create_univariate_summary,
    create_data_quality_report,
    create_correlation_analysis
)

# Importar librerías necesarias para backward compatibility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Configurar warnings
warnings.filterwarnings('ignore')

# Función de ayuda para análisis completo
def run_complete_analysis(df, target_column=None):
    """
    Ejecuta un análisis completo del DataFrame
    
    Args:
        df: DataFrame a analizar
        target_column: Columna objetivo (opcional)
        
    Returns:
        Diccionario con resultados del análisis
    """
    print("="*100)
    print("ANÁLISIS COMPLETO DEL DATASET")
    print("="*100)
    
    # 1. Análisis de calidad de datos
    print("\n1. ANÁLISIS DE CALIDAD DE DATOS")
    print("-" * 50)
    quality_report = create_data_quality_report(df)
    
    # 2. Análisis univariado básico
    print("\n2. ANÁLISIS UNIVARIADO")
    print("-" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        print(f"Analizando {len(numeric_cols)} variables numéricas...")
        # Tomar una muestra para no sobrecargar
        sample_numeric = numeric_cols[:5]
        run_univariate_analysis(df, continuous_cols=sample_numeric)
    
    if len(categorical_cols) > 0:
        print(f"Analizando {len(categorical_cols)} variables categóricas...")
        # Tomar una muestra para no sobrecargar
        sample_categorical = categorical_cols[:3]
        run_univariate_analysis(df, categorical_cols=sample_categorical)
    
    # 3. Análisis de correlaciones
    print("\n3. ANÁLISIS DE CORRELACIONES")
    print("-" * 50)
    correlation_matrix = create_correlation_analysis(df, target_column)
    
    # 4. Dashboard avanzado
    print("\n4. DASHBOARD AVANZADO")
    print("-" * 50)
    create_advanced_univariate_dashboard(df)
    
    # 5. Resumen consolidado
    print("\n5. RESUMEN CONSOLIDADO")
    print("-" * 50)
    summary_results = create_univariate_summary(df)
    
    return {
        'quality_report': quality_report,
        'correlation_matrix': correlation_matrix,
        'summary_results': summary_results,
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols
    }


# Función para análisis rápido
def quick_analysis(df, columns=None, max_cols=5):
    """
    Ejecuta un análisis rápido de las columnas especificadas
    
    Args:
        df: DataFrame a analizar
        columns: Lista de columnas a analizar (opcional)
        max_cols: Número máximo de columnas a analizar
        
    Returns:
        Diccionario con resultados del análisis
    """
    if columns is None:
        # Seleccionar automáticamente las columnas más relevantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Tomar una muestra
        columns = numeric_cols[:max_cols//2] + categorical_cols[:max_cols//2]
    
    results = {}
    
    for col in columns:
        print(f"\n{'='*60}")
        print(f"ANÁLISIS RÁPIDO: {col.upper()}")
        print(f"{'='*60}")
        
        if df[col].dtype in ['int64', 'float64']:
            results[col] = enhanced_univariate_analysis(df, col, 'numeric')
        else:
            results[col] = enhanced_univariate_analysis(df, col, 'categorical')
    
    return results


# Función para obtener recomendaciones
def get_preprocessing_recommendations(df):
    """
    Obtiene recomendaciones de preprocesamiento basadas en el análisis
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con recomendaciones
    """
    recommendations = {
        'missing_data': [],
        'outliers': [],
        'transformations': [],
        'encoding': [],
        'scaling': []
    }
    
    # Análisis de datos faltantes
    missing_data = df.isnull().sum()
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            if missing_pct > 50:
                recommendations['missing_data'].append(f"Considerar eliminar columna '{col}' (>{missing_pct:.1f}% faltante)")
            elif missing_pct > 10:
                recommendations['missing_data'].append(f"Imputar valores faltantes en '{col}' ({missing_pct:.1f}% faltante)")
    
    # Análisis de outliers y transformaciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # Asimetría
        skew = df[col].skew()
        if abs(skew) > 2:
            recommendations['transformations'].append(f"Aplicar transformación logarítmica a '{col}' (skew={skew:.2f})")
        
        # Outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > len(df) * 0.05:
            recommendations['outliers'].append(f"Tratar outliers en '{col}' ({len(outliers)} valores, {len(outliers)/len(df)*100:.1f}%)")
    
    # Análisis de variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
            recommendations['encoding'].append(f"Considerar agrupación de categorías en '{col}' ({unique_count} categorías)")
        elif unique_count > 10:
            recommendations['encoding'].append(f"Aplicar encoding adecuado a '{col}' ({unique_count} categorías)")
    
    # Recomendaciones de escalado
    if len(numeric_cols) > 1:
        # Verificar rangos muy diferentes
        ranges = {}
        for col in numeric_cols:
            col_range = df[col].max() - df[col].min()
            ranges[col] = col_range
        
        max_range = max(ranges.values())
        min_range = min(ranges.values())
        
        if max_range / min_range > 100:
            recommendations['scaling'].append("Considerar estandarización/normalización (rangos muy diferentes)")
    
    return recommendations


# Función para generar reporte HTML (placeholder)
def generate_html_report(df, output_path="analysis_report.html"):
    """
    Genera un reporte HTML con el análisis completo
    
    Args:
        df: DataFrame a analizar
        output_path: Ruta donde guardar el reporte
        
    Returns:
        Ruta del archivo generado
    """
    print("Función de generación de reporte HTML en desarrollo...")
    print("Por ahora, use las funciones individuales para análisis específicos.")
    return None


# Configuración de estilos para matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
