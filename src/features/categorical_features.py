"""
Módulo para la codificación de variables categóricas.
"""
import pandas as pd
from typing import List

def apply_onehot_encoding(df: pd.DataFrame, categorical_cols: List[str], drop_first: bool = True) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a variables de baja cardinalidad.
    """
    df_result = df.copy()
    for col in categorical_cols:
        if col in df_result.columns:
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first, dtype=float)
            df_result = pd.concat([df_result, dummies], axis=1)
    return df_result

def apply_target_encoding(df: pd.DataFrame, categorical_col: str, target_col: str, smoothing: float = 10.0) -> pd.DataFrame:
    """
    Aplica Target Encoding con suavizado.
    """
    df_result = df.copy()
    global_mean = df_result[target_col].mean()
    
    stats = df_result.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    
    smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    df_result[f'{categorical_col}_target_encoded'] = df_result[categorical_col].map(smoothed_mean)
    df_result[f'{categorical_col}_target_encoded'].fillna(global_mean, inplace=True)
    
    return df_result

def group_rare_categories(df: pd.DataFrame, categorical_col: str, threshold: float = 0.01, other_label: str = 'Other') -> pd.DataFrame:
    """
    Agrupa categorías poco frecuentes en 'Other'.
    """
    df_result = df.copy()
    freqs = df_result[categorical_col].value_counts(normalize=True)
    rare_cats = freqs[freqs < threshold].index
    
    new_col_name = f'{categorical_col}_grouped'
    df_result[new_col_name] = df_result[categorical_col].copy()
    df_result.loc[df_result[categorical_col].isin(rare_cats), new_col_name] = other_label
    
    return df_result
