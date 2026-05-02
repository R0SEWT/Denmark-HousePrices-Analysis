"""
Módulo para la creación de variables derivadas de tamaño.
"""
import pandas as pd

def create_size_derived_features(df: pd.DataFrame, sqm_col: str = 'sqm', rooms_col: str = 'no_rooms') -> pd.DataFrame:
    """
    Crea variables derivadas del tamaño.
    """
    df_result = df.copy()
    df_result['sqm_per_room'] = df_result[sqm_col] / df_result[rooms_col].replace(0, 1) # Evitar división por cero
    return df_result
