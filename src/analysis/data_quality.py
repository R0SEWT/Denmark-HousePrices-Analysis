"""
Módulo de calidad de datos
Contiene funciones para analizar la calidad de los datos: valores nulos, duplicados, etc.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_df_null_resume_and_percentages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtiene un resumen de los valores nulos en el DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Tupla con (df_nulls, df_nulls_resume)
    """
    df_nulls = df[df.isna().any(axis=1)]
    
    df_nulls_resume = (df.isna().sum() / df.shape[0] * 100)
    df_nulls_resume = df_nulls_resume[df_nulls_resume > 0].sort_values(ascending=False).reset_index()
    df_nulls_resume.columns = ["column", "null_percentage"]
    df_nulls_resume["null_percentage"] = df_nulls_resume["null_percentage"].apply(lambda x: f"{x:.2f} %")
    return df_nulls, df_nulls_resume


def get_duplicate_percentage(df: pd.DataFrame) -> float:
    """
    Calcula el porcentaje de filas duplicadas en el DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Porcentaje de duplicados
    """
    return round((df.duplicated().sum() / df.shape[0]) * 100, 2)


def plot_null_heatmap(df: pd.DataFrame):
    """
    Crea un mapa de calor de los valores nulos.
    
    Args:
        df: DataFrame a visualizar
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title("Mapa de valores nulos")
    plt.show()


def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene información sobre los tipos de columnas.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        DataFrame con información de columnas
    """
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum()
    })


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Análisis completo de calidad de datos.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con métricas de calidad
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_percentage': get_duplicate_percentage(df),
        'null_info': get_df_null_resume_and_percentages(df)[1],
        'column_info': get_column_types(df),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return quality_metrics


def verify_duplicates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Verifica si hay filas duplicadas basadas en columnas específicas.
    
    Args:
        df: DataFrame a analizar
        columns: Lista de nombres de columnas para verificar duplicados
        
    Returns:
        DataFrame con filas duplicadas
    """
    # Convertir a lista si es un Index de pandas
    if hasattr(columns, 'tolist'):
        columns = columns.tolist()
    elif not isinstance(columns, list):
        columns = list(columns)
    
    # Verificar duplicados
    duplicated_mask = df.duplicated(subset=columns, keep=False)
    
    if duplicated_mask.sum() > 0:
        # Solo ordenar si hay duplicados para evitar errores
        df_duplicated = df[duplicated_mask].sort_values(by=columns)
        return df_duplicated.shape[0]
    else:
        return 0



