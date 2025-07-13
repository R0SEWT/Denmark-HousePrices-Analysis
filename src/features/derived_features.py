"""
Módulo para la creación de variables derivadas de precio y tamaño.
"""
import pandas as pd
import numpy as np

def create_price_derived_features(df: pd.DataFrame, price_col: str = 'purchase_price') -> pd.DataFrame:
    """
    Crea variables derivadas del precio.
    """
    df_result = df.copy()
    df_result['log_price'] = np.log1p(df_result[price_col])
    
    if 'region' in df_result.columns:
        regional_median = df_result.groupby('region')[price_col].transform('median')
        df_result['price_ratio_regional_median'] = df_result[price_col] / regional_median
    
    return df_result

def create_size_derived_features(df: pd.DataFrame, sqm_col: str = 'sqm', rooms_col: str = 'no_rooms') -> pd.DataFrame:
    """
    Crea variables derivadas del tamaño.
    """
    df_result = df.copy()
    df_result['sqm_per_room'] = df_result[sqm_col] / df_result[rooms_col].replace(0, 1) # Evitar división por cero
    return df_result
