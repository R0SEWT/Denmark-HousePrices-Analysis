"""
Módulo de visualización
Contiene funciones especializadas para crear visualizaciones
para análisis y modelado de datos inmobiliarios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple


def plot_target_distribution(df: pd.DataFrame, target: str):
    """
    Grafica la distribución de la variable objetivo.
    
    Args:
        df: DataFrame
        target: Nombre de la columna objetivo
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(df[target].dropna(), kde=True)
    plt.title(f"Distribución de {target}")
    plt.xlabel(target)
    plt.show()


def create_correlation_heatmap(df: pd.DataFrame, figsize=(12, 8)):
    """
    Crea un mapa de calor de correlaciones.
    
    Args:
        df: DataFrame
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Mapa de Correlaciones')
    plt.tight_layout()
    plt.show()


def create_distribution_comparison(df: pd.DataFrame, columns: list, ncols=2):
    """
    Crea una comparación de distribuciones para múltiples columnas.
    
    Args:
        df: DataFrame
        columns: Lista de columnas a comparar
        ncols: Número de columnas en la grilla
    """
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
    
    # Ocultar ejes vacíos
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def create_boxplot_comparison(df: pd.DataFrame, columns: list, ncols=2):
    """
    Crea boxplots para comparar múltiples columnas.
    
    Args:
        df: DataFrame
        columns: Lista de columnas a comparar
        ncols: Número de columnas en la grilla
    """
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.boxplot(y=df[col].dropna(), ax=axes[i])
            axes[i].set_title(f'Boxplot de {col}')
            axes[i].set_ylabel(col)
    
    # Ocultar ejes vacíos
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def create_categorical_summary_plot(df: pd.DataFrame, categorical_columns: list, max_categories=10):
    """
    Crea un resumen visual de variables categóricas.
    
    Args:
        df: DataFrame
        categorical_columns: Lista de columnas categóricas
        max_categories: Número máximo de categorías a mostrar por variable
    """
    n_cols = len(categorical_columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 6*n_cols))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(categorical_columns):
        # Obtener las top categorías
        top_cats = df[col].value_counts().head(max_categories)
        
        # Crear gráfico de barras
        sns.barplot(x=top_cats.values, y=top_cats.index, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Top {max_categories} categorías de {col}')
        axes[i].set_xlabel('Frecuencia')
        
        # Añadir etiquetas de porcentaje
        total = len(df[col].dropna())
        for j, (idx, val) in enumerate(top_cats.items()):
            axes[i].text(val + total * 0.01, j, f'{val/total*100:.1f}%', 
                        va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def create_missing_data_visualization(df: pd.DataFrame):
    """
    Crea visualizaciones para datos faltantes.
    
    Args:
        df: DataFrame
    """
    # Calcular porcentajes de datos faltantes
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    
    if len(missing_percentage) == 0:
        print("No hay datos faltantes en el DataFrame")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras del porcentaje de datos faltantes
    missing_percentage.plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_title('Porcentaje de Datos Faltantes por Columna')
    axes[0].set_ylabel('Porcentaje (%)')
    axes[0].set_xlabel('Columnas')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Mapa de calor de datos faltantes
    missing_subset = df[missing_percentage.index].isnull()
    sns.heatmap(missing_subset.T, cbar=True, ax=axes[1], 
                cmap='YlOrRd', yticklabels=True, xticklabels=False)
    axes[1].set_title('Patrón de Datos Faltantes')
    axes[1].set_xlabel('Observaciones')
    
    plt.tight_layout()
    plt.show()


def create_outlier_visualization(df: pd.DataFrame, numeric_columns: list):
    """
    Crea visualizaciones para detectar outliers.
    
    Args:
        df: DataFrame
        numeric_columns: Lista de columnas numéricas
    """
    n_cols = min(len(numeric_columns), 4)  # Máximo 4 columnas por fila
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_columns):
        if i < len(axes):
            sns.boxplot(y=df[col].dropna(), ax=axes[i])
            axes[i].set_title(f'Outliers en {col}')
            axes[i].set_ylabel(col)
    
    # Ocultar ejes vacíos
    for i in range(len(numeric_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def create_advanced_univariate_dashboard(df):
    """
    Crea un dashboard avanzado con resumen univariado
    
    Args:
        df: DataFrame a analizar
    """
    print("="*80)
    print("DASHBOARD UNIVARIADO AVANZADO")
    print("="*80)
    
    # Obtener tipos de columnas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nRESUMEN GENERAL:")
    print(f"   • Variables numéricas: {len(numeric_cols)}")
    print(f"   • Variables categóricas: {len(categorical_cols)}")
    print(f"   • Total de observaciones: {len(df):,}")
    
    # 1. Visualización de datos faltantes
    print(f"\n1. ANÁLISIS DE DATOS FALTANTES")
    print("-" * 50)
    create_missing_data_visualization(df)
    
    # 2. Distribuciones numéricas
    if numeric_cols:
        print(f"\n2. DISTRIBUCIONES NUMÉRICAS")
        print("-" * 50)
        create_distribution_comparison(df, numeric_cols[:8])  # Máximo 8 para no sobrecargar
        
        print(f"\n3. DETECCIÓN DE OUTLIERS")
        print("-" * 50)
        create_outlier_visualization(df, numeric_cols[:8])
        
        print(f"\n4. CORRELACIONES")
        print("-" * 50)
        create_correlation_heatmap(df[numeric_cols])
    
    # 3. Variables categóricas
    if categorical_cols:
        print(f"\n5. VARIABLES CATEGÓRICAS")
        print("-" * 50)
        create_categorical_summary_plot(df, categorical_cols[:5])  # Máximo 5 para no sobrecargar
    
    print(f"\n{'='*80}")


def plot_feature_importance(
    features: List[str],
    scores: List[float],
    output_path: Optional[Path] = None,
    title: str = 'Variables Predictivas Críticas para el Modelado',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Crea una visualización profesional de importancia de features.
    
    Args:
        features: Lista con nombres de las features
        scores: Lista con las puntuaciones de importancia
        output_path: Ruta donde guardar la visualización (opcional)
        title: Título para la visualización
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Figura de matplotlib
    """
    # Crear figura
    fig = plt.figure(figsize=figsize)
    
    # Crear gráfico con estilo corporativo
    plt.style.use('seaborn-v0_8-whitegrid')
    bars = plt.barh(np.arange(len(features)), scores, color='#3366cc', alpha=0.8)
    
    # Añadir valores a las barras
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', ha='left', fontsize=10, color='#333333')
    
    plt.yticks(np.arange(len(features)), features, fontsize=11)
    plt.xlabel('Puntuación de Importancia Combinada', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Guardar la visualización si se especifica ruta
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    model_names: List[str],
    metric_values: Dict[str, List[float]],
    metric_name: str = 'R²',
    ascending: bool = True,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Crea un gráfico de barras para comparar rendimiento de múltiples modelos.
    
    Args:
        model_names: Lista con nombres de modelos
        metric_values: Diccionario con valores de métricas por modelo
        metric_name: Nombre de la métrica para etiquetas
        ascending: Si es True, ordena los valores ascendentemente
        output_path: Ruta para guardar la visualización (opcional)
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    # Extraer valores de la métrica principal
    values = metric_values.get(list(metric_values.keys())[0], [])
    
    # Ordenar por rendimiento
    sorted_indices = np.argsort(values)
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    
    sorted_models = [model_names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Crear figura
    fig = plt.figure(figsize=figsize)
    
    # Crear gráfico de barras
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_models)))
    bars = plt.bar(sorted_models, sorted_values, color=colors, alpha=0.8)
    
    # Añadir valores a las barras
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        plt.text(bar.get_x() + bar.get_width()/2, value + max(sorted_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel(f'Valor de {metric_name}', fontsize=12)
    plt.title(f'Comparación de Modelos por {metric_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Guardar la visualización si se especifica ruta
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
    print("DASHBOARD COMPLETADO")
    print("="*80)
