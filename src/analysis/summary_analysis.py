"""
Módulo de análisis consolidado
Contiene funciones para crear resúmenes y análisis consolidados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_univariate_summary(df, analyses_results=None):
    """
    Crea un resumen consolidado del análisis univariado con insights de negocio
    
    Args:
        df: DataFrame a analizar
        analyses_results: Resultados de análisis previos (opcional)
        
    Returns:
        Diccionario con métricas del resumen
    """    
    print("="*100)
    print("RESUMEN CONSOLIDADO DEL ANÁLISIS UNIVARIADO")
    print("="*100)
    
    # Análisis general del dataset
    print(f"\nPANORAMA GENERAL DEL DATASET:")
    print(f"   • Total de observaciones: {len(df):,}")
    print(f"   • Total de variables: {len(df.columns)}")
    print(f"   • Variables numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   • Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Calidad de datos
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    print(f"\nCALIDAD DE DATOS:")
    print(f"   • Variables con valores faltantes: {len(missing_cols)}")
    print(f"   • Porcentaje total de datos faltantes: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    # Análisis de variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nANÁLISIS DE VARIABLES NUMÉRICAS:")
    
    # Estadísticas de asimetría
    high_skew_vars = []
    transformation_needed = []
    
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            high_skew_vars.append((col, skew))
        if abs(skew) > 2:
            transformation_needed.append((col, skew))
    
    print(f"   • Variables con alta asimetría (|skew| > 1): {len(high_skew_vars)}")
    for var, skew in high_skew_vars[:5]:  # Top 5
        print(f"     → {var}: {skew:.2f}")
    
    print(f"   • Variables que requieren transformación (|skew| > 2): {len(transformation_needed)}")
    for var, skew in transformation_needed:
        print(f"     → {var}: {skew:.2f}")
    
    # Análisis de outliers
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)
    
    high_outlier_vars = [(k, v) for k, v in outlier_summary.items() if v > len(df) * 0.05]
    print(f"\nDETECCIÓN DE OUTLIERS:")
    print(f"   • Variables con >5% de outliers: {len(high_outlier_vars)}")
    for var, count in high_outlier_vars:
        print(f"     → {var}: {count:,} outliers ({count/len(df)*100:.1f}%)")
    
    # Análisis de variables categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nANÁLISIS DE VARIABLES CATEGÓRICAS:")
    
    high_cardinality = []
    low_diversity = []
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
            high_cardinality.append((col, unique_count))
        
        # Cálculo de diversidad Shannon
        proportions = df[col].value_counts(normalize=True)
        shannon_div = -sum(proportions * np.log(proportions))
        if shannon_div < 1:
            low_diversity.append((col, shannon_div))
    
    print(f"   • Variables con alta cardinalidad (>50 categorías): {len(high_cardinality)}")
    for var, count in high_cardinality:
        print(f"     → {var}: {count:,} categorías únicas")
    
    print(f"   • Variables con baja diversidad (Shannon < 1): {len(low_diversity)}")
    for var, div in low_diversity:
        print(f"     → {var}: {div:.3f}")
    
    # Insights de negocio específicos
    _print_business_insights(df)
    
    # Recomendaciones estratégicas
    _print_strategic_recommendations(df, transformation_needed, high_outlier_vars, high_cardinality)
    
    return {
        'total_observations': len(df),
        'numeric_variables': len(numeric_cols),
        'categorical_variables': len(df.select_dtypes(include=['object']).columns),
        'high_skew_vars': len(high_skew_vars),
        'transformation_needed': len(transformation_needed),
        'high_outlier_vars': len(high_outlier_vars),
        'high_cardinality': len(high_cardinality),
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }


def _print_business_insights(df):
    """Imprime insights de negocio específicos"""
    print(f"\nINSIGHTS DE NEGOCIO CLAVE:")
    
    # Análisis de precios
    if 'purchase_price' in df.columns:
        price_median = df['purchase_price'].median()
        price_mean = df['purchase_price'].mean()
        price_std = df['purchase_price'].std()
        cv_price = (price_std / price_mean) * 100
        
        print(f"   • PRECIOS DE VIVIENDAS:")
        print(f"     → Mediana: ${price_median:,.0f}")
        print(f"     → Media: ${price_mean:,.0f}")
        print(f"     → Coeficiente de variación: {cv_price:.1f}%")
        
        if cv_price > 50:
            print(f"     → Alta variabilidad en precios - mercado diversificado")
        
        # Análisis de precios por m2
        if 'sqm_price' in df.columns:
            sqm_price_median = df['sqm_price'].median()
            print(f"     → Precio mediano por m²: ${sqm_price_median:,.0f}")
    
    # Análisis de características físicas
    if 'sqm' in df.columns:
        sqm_median = df['sqm'].median()
        sqm_q75 = df['sqm'].quantile(0.75)
        print(f"   • CARACTERÍSTICAS FÍSICAS:")
        print(f"     → Tamaño mediano: {sqm_median:.0f} m²")
        print(f"     → 75% de propiedades ≤ {sqm_q75:.0f} m²")
    
    if 'no_rooms' in df.columns:
        rooms_mode = df['no_rooms'].mode().iloc[0]
        rooms_median = df['no_rooms'].median()
        print(f"     → Número de habitaciones más común: {rooms_mode}")
        print(f"     → Mediana de habitaciones: {rooms_median}")
    
    # Análisis geográfico
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()
        top_region = region_counts.index[0]
        top_region_pct = region_counts.iloc[0] / len(df) * 100
        print(f"   • DISTRIBUCIÓN GEOGRÁFICA:")
        print(f"     → Región dominante: {top_region} ({top_region_pct:.1f}%)")
        print(f"     → Número de regiones: {df['region'].nunique()}")
    
    # Análisis temporal
    if 'year_build' in df.columns:
        year_median = df['year_build'].median()
        year_q25 = df['year_build'].quantile(0.25)
        year_q75 = df['year_build'].quantile(0.75)
        print(f"   • CARACTERÍSTICAS TEMPORALES:")
        print(f"     → Año de construcción mediano: {year_median:.0f}")
        print(f"     → 50% de propiedades construidas entre {year_q25:.0f} y {year_q75:.0f}")


def _print_strategic_recommendations(df, transformation_needed, high_outlier_vars, high_cardinality):
    """Imprime recomendaciones estratégicas"""
    print(f"\nRECOMENDACIONES ESTRATÉGICAS:")
    
    print(f"   • PREPROCESSING:")
    if transformation_needed:
        print(f"     → Aplicar transformaciones logarítmicas a variables con alta asimetría")
    if high_outlier_vars:
        print(f"     → Implementar técnicas de tratamiento de outliers (winsorizing, capping)")
    if high_cardinality:
        print(f"     → Considerar agrupación de categorías para variables de alta cardinalidad")
    
    print(f"   • MODELADO:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if any(abs(df[col].skew()) > 2 for col in numeric_cols):
        print(f"     → Considerar modelos robustos a outliers (Random Forest, Gradient Boosting)")
    print(f"     → Aplicar técnicas de feature engineering para variables categóricas")
    print(f"     → Considerar interacciones entre variables geográficas y de precio")
    
    print(f"   • NEGOCIO:")
    if 'purchase_price' in df.columns and df['purchase_price'].std() / df['purchase_price'].mean() > 0.5:
        print(f"     → Segmentar mercado por rangos de precio para análisis específicos")
    print(f"     → Considerar análisis geográfico detallado para pricing estratégico")
    print(f"     → Implementar análisis temporal para identificar tendencias estacionales")


def create_data_quality_report(df):
    """
    Crea un reporte completo de calidad de datos
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con métricas de calidad
    """
    print("="*80)
    print("REPORTE DE CALIDAD DE DATOS")
    print("="*80)
    
    # Métricas básicas
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"\nMÉTRICAS GENERALES:")
    print(f"   • Total de celdas: {total_cells:,}")
    print(f"   • Celdas faltantes: {missing_cells:,}")
    print(f"   • Porcentaje faltante: {missing_percentage:.2f}%")
    
    # Análisis por columna
    print(f"\nANÁLISIS POR COLUMNA:")
    missing_by_column = df.isnull().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
    
    if len(missing_by_column) > 0:
        print(f"   • Columnas con datos faltantes: {len(missing_by_column)}")
        for col, count in missing_by_column.head(10).items():
            pct = (count / len(df)) * 100
            print(f"     → {col}: {count:,} ({pct:.1f}%)")
    else:
        print("   • No hay datos faltantes")
    
    # Análisis de duplicados
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100
    
    print(f"\nANÁLISIS DE DUPLICADOS:")
    print(f"   • Filas duplicadas: {duplicates:,}")
    print(f"   • Porcentaje duplicado: {duplicate_percentage:.2f}%")
    
    # Análisis de tipos de datos
    print(f"\nANÁLISIS DE TIPOS DE DATOS:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} columnas")
    
    return {
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_percentage': missing_percentage,
        'columns_with_missing': len(missing_by_column),
        'duplicate_rows': duplicates,
        'duplicate_percentage': duplicate_percentage,
        'data_types': dtype_counts.to_dict()
    }


def create_correlation_analysis(df, target_column=None):
    """
    Crea un análisis de correlaciones
    
    Args:
        df: DataFrame
        target_column: Columna objetivo para análisis específico
        
    Returns:
        Matriz de correlaciones
    """
    print("="*80)
    print("ANÁLISIS DE CORRELACIONES")
    print("="*80)
    
    # Obtener solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        print("   • No hay suficientes variables numéricas para análisis de correlación")
        return None
    
    # Calcular matriz de correlación
    correlation_matrix = numeric_df.corr()
    
    # Correlaciones más altas
    print(f"\nCORRELACIONES MÁS ALTAS:")
    
    # Obtener correlaciones sin la diagonal
    correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            if not np.isnan(corr):
                correlation_pairs.append((col1, col2, corr))
    
    # Ordenar por correlación absoluta
    correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"   • Top 10 correlaciones:")
    for i, (col1, col2, corr) in enumerate(correlation_pairs[:10]):
        print(f"     {i+1}. {col1} - {col2}: {corr:.3f}")
    
    # Análisis con variable objetivo si se especifica
    if target_column and target_column in numeric_df.columns:
        print(f"\nCORRELACIONES CON {target_column.upper()}:")
        target_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)
        target_correlations = target_correlations.drop(target_column)  # Remover auto-correlación
        
        print(f"   • Top 10 correlaciones con {target_column}:")
        for i, (col, corr) in enumerate(target_correlations.head(10).items()):
            print(f"     {i+1}. {col}: {corr:.3f}")
    
    # Crear visualización
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix
