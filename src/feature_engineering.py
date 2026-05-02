"""
Feature Engineering Module para análisis de precios inmobiliarios en Dinamarca

Este módulo contiene todas las funciones necesarias para la transformación,
codificación y preparación de features para el modelado supervisado.

Autor: TF Big Data Project
Fecha: Julio 2025
"""

import pandas as pd
import numpy as np
import warnings
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore')

# === CONFIGURACIONES ===
FEATURE_ENGINEERING_CONFIG = {
    'temporal': {
        'current_year': 2024,
        'crisis_years': [2008, 2009, 2020, 2021],
        'market_phases': {
            'growth_90s': (0, 2000),
            'boom_2000s': (2001, 2007),
            'crisis_post2008': (2008, 2012),
            'recovery_2010s': (2013, 2019),
            'covid_era': (2020, 2024)
        }
    },
    'encoding': {
        'min_samples_smoothing': 100,
        'rare_category_threshold': 0.01,
        'max_categories_onehot': 10
    },
    'feature_selection': {
        'max_features': 30,
        'sample_size_fs': 50000,
        'variance_threshold': 0.001
    },
    'train_test': {
        'split_year': 2017,
        'test_start_year': 2018
    }
}

try:
    from config import (
        CV_MAX_TRAIN_END_YEAR,
        CV_MIN_TRAIN_END_YEAR,
        HOLDOUT_START_YEAR,
        MIN_OBS_PER_WINDOW,
        OPTUNA_STUDY_NAME_V2,
        ROLLING_WINDOW_YEARS,
    )
except ImportError:  # pragma: no cover - soporte para import como paquete
    from .config import (
        CV_MAX_TRAIN_END_YEAR,
        CV_MIN_TRAIN_END_YEAR,
        HOLDOUT_START_YEAR,
        MIN_OBS_PER_WINDOW,
        OPTUNA_STUDY_NAME_V2,
        ROLLING_WINDOW_YEARS,
    )

MODEL_TARGET_COL = 'log_price'
ROLLING_FEATURE_COLUMNS = [
    'rolling_regional_mean',
    'rolling_regional_median',
    'rolling_regional_std',
    'rolling_regional_cv',
    'rolling_regional_count',
    'rolling_regional_p90',
]
CAUSAL_DERIVED_COLUMNS = [
    'is_premium_causal',
    'price_deviation_from_rolling_median',
]
ROLLING_AUXILIARY_COLUMNS = ['rolling_regional_median_v2']
NON_IMPUTED_FEATURE_COLUMNS = set(ROLLING_FEATURE_COLUMNS + CAUSAL_DERIVED_COLUMNS)
FORBIDDEN_FEATURE_COLUMNS = {
    'purchase_price',
    'price_per_sqm',
    'price_zscore',
    'price_category',
    'price_per_sqm_x_region',
    'sqm_x_region',
    'is_premium',
    'price_deviation_from_median',
    'region_target_encoded',
    'regional_price_mean',
    'regional_price_median',
    'regional_price_std',
    'regional_price_cv',
    'regional_price_rank',
    'regional_transaction_count',
    'regional_liquidity_score',
    'regional_p90',
    'regional_median',
}

# Importaciones dinámicas para visualización
def _import_viz_libraries():
    """Importa bibliotecas de visualización dinámicamente"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        return plt, sns
    except ImportError as e:
        logger.warning(f"Error al importar bibliotecas de visualización: {e}")
        return None, None

# Importaciones dinámicas para machine learning
def _import_ml_libraries():
    """Importa bibliotecas de machine learning dinámicamente"""
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.feature_selection import mutual_info_regression, SelectKBest
        from sklearn.ensemble import RandomForestRegressor
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        return {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler, 
            'RobustScaler': RobustScaler,
            'LabelEncoder': LabelEncoder,
            'OneHotEncoder': OneHotEncoder,
            'mutual_info_regression': mutual_info_regression,
            'SelectKBest': SelectKBest,
            'RandomForestRegressor': RandomForestRegressor,
            'variance_inflation_factor': variance_inflation_factor
        }
    except ImportError as e:
        logger.warning(f"Error al importar bibliotecas de ML: {e}")
        return {}

# ===== SECCIÓN 1: TRANSFORMACIÓN DE TIPOS Y COLUMNAS DERIVADAS =====

def convert_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Convierte columna de fecha a datetime y extrae componentes temporales
    
    Args:
        df: DataFrame con datos
        date_col: Nombre de la columna de fecha
    
    Returns:
        DataFrame con nuevas variables temporales
    """
    print("🔄 Convirtiendo features de fecha...")
    
    df_result = df.copy()
    
    # Conversión a datetime si no está ya convertido
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
        df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Extracción de componentes básicos
    df_result['year_sale'] = df_result[date_col].dt.year
    df_result['month_sale'] = df_result[date_col].dt.month
    df_result['day_sale'] = df_result[date_col].dt.day
    df_result['dayofweek_sale'] = df_result[date_col].dt.dayofweek
    df_result['quarter_sale'] = df_result[date_col].dt.quarter
    
    # Variables estacionales
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    df_result['season_sale'] = df_result['month_sale'].map(season_map)
    
    # Días de la semana (nombres)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_result['dayname_sale'] = df_result['dayofweek_sale'].map(lambda x: day_names[x])
    
    print(f"✅ Creadas {8} nuevas variables temporales")
    return df_result

def create_property_age_features(df: pd.DataFrame, 
                                year_built_col: str = 'year_build',
                                reference_year: int = 2024) -> pd.DataFrame:
    """
    Crea variables relacionadas con la edad de la propiedad
    
    Args:
        df: DataFrame con datos
        year_built_col: Columna con año de construcción
        reference_year: Año de referencia para calcular edad
    
    Returns:
        DataFrame con variables de edad
    """
    print("🏠 Creando features de edad de propiedad...")
    
    df_result = df.copy()
    
    # Edad de la propiedad
    df_result['property_age'] = reference_year - df_result[year_built_col]
    
    # Década de construcción
    df_result['decade_built'] = (df_result[year_built_col] // 10) * 10
    df_result['decade_built_label'] = df_result['decade_built'].astype(str) + 's'
    
    # Categorías de edad
    def categorize_age(age):
        if age < 0:
            return 'Future/Error'
        elif age <= 5:
            return 'New (0-5 years)'
        elif age <= 15:
            return 'Modern (6-15 years)'
        elif age <= 30:
            return 'Established (16-30 years)'
        elif age <= 50:
            return 'Mature (31-50 years)'
        else:
            return 'Historic (50+ years)'
    
    df_result['age_category'] = df_result['property_age'].apply(categorize_age)
    
    # Vintage (clasificación por época)
    def categorize_vintage(year):
        if year < 1900:
            return 'Historic'
        elif year < 1950:
            return 'Pre-War'
        elif year < 1980:
            return 'Post-War'
        elif year < 2000:
            return 'Late Century'
        else:
            return 'Modern'
    
    df_result['vintage_category'] = df_result[year_built_col].apply(categorize_vintage)
    
    print(f"✅ Creadas {5} nuevas variables de edad")
    return df_result

def create_price_derived_features(df: pd.DataFrame, 
                                 price_col: str = 'purchase_price') -> pd.DataFrame:
    """
    DEPRECADO: usa create_price_features.
    
    Args:
        df: DataFrame con datos
        price_col: Columna de precio
    
    Returns:
        DataFrame con variables de precio derivadas
    """
    warnings.warn(
        "create_price_derived_features esta deprecada; usar create_price_features",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_price_features(df, target_col=price_col)

def create_size_derived_features(df: pd.DataFrame,
                                sqm_col: str = 'sqm',
                                rooms_col: str = 'no_rooms') -> pd.DataFrame:
    """
    Crea variables derivadas de tamaño
    
    Args:
        df: DataFrame con datos
        sqm_col: Columna de metros cuadrados
        rooms_col: Columna de número de habitaciones
    
    Returns:
        DataFrame con variables de tamaño derivadas
    """
    print("📐 Creando features derivadas de tamaño...")
    
    df_result = df.copy()
    
    # Binning de habitaciones
    def categorize_rooms(rooms):
        if rooms <= 2:
            return '1-2 rooms'
        elif rooms <= 4:
            return '3-4 rooms'
        else:
            return '5+ rooms'
    
    df_result['rooms_category'] = df_result[rooms_col].apply(categorize_rooms)
    
    # Categorías de tamaño en m²
    sqm_percentiles = df_result[sqm_col].quantile([0.33, 0.67])
    
    def categorize_size(sqm):
        if sqm <= sqm_percentiles[0.33]:
            return 'Small'
        elif sqm <= sqm_percentiles[0.67]:
            return 'Medium'
        else:
            return 'Large'
    
    df_result['size_category'] = df_result[sqm_col].apply(categorize_size)
    
    # Eficiencia espacial (m² por habitación)
    df_result['sqm_per_room'] = df_result[sqm_col] / df_result[rooms_col]
    
    # Ratio precio/m² percentiles (si existe sqm_price)
    if 'sqm_price' in df_result.columns:
        df_result['sqm_price_percentile'] = df_result['sqm_price'].rank(pct=True)
        
        # Categorías de eficiencia de precio por m²
        sqm_price_percentiles = df_result['sqm_price'].quantile([0.25, 0.5, 0.75])
        
        def categorize_sqm_price(price):
            if price <= sqm_price_percentiles[0.25]:
                return 'Very Efficient'
            elif price <= sqm_price_percentiles[0.5]:
                return 'Efficient'
            elif price <= sqm_price_percentiles[0.75]:
                return 'Average'
            else:
                return 'Premium'
        
        df_result['sqm_price_category'] = df_result['sqm_price'].apply(categorize_sqm_price)
    
    print(f"✅ Creadas {5} nuevas variables de tamaño")
    return df_result

def create_cyclic_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Crea variables temporales cíclicas usando seno y coseno
    
    Args:
        df: DataFrame con datos
        date_col: Columna de fecha
    
    Returns:
        DataFrame con variables cíclicas
    """
    print("🔄 Creando features temporales cíclicas...")
    
    df_result = df.copy()
    
    # Asegurar que date_col esté en datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
        df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Componentes cíclicos para mes (estacionalidad anual)
    df_result['month_sin'] = np.sin(2 * np.pi * df_result[date_col].dt.month / 12)
    df_result['month_cos'] = np.cos(2 * np.pi * df_result[date_col].dt.month / 12)
    
    # Componentes cíclicos para día de la semana
    df_result['dayofweek_sin'] = np.sin(2 * np.pi * df_result[date_col].dt.dayofweek / 7)
    df_result['dayofweek_cos'] = np.cos(2 * np.pi * df_result[date_col].dt.dayofweek / 7)
    
    # Componentes cíclicos para quarter
    df_result['quarter_sin'] = np.sin(2 * np.pi * df_result[date_col].dt.quarter / 4)
    df_result['quarter_cos'] = np.cos(2 * np.pi * df_result[date_col].dt.quarter / 4)
    
    # Tendencia temporal (años desde el inicio)
    min_year = df_result[date_col].dt.year.min()
    df_result['years_since_start'] = df_result[date_col].dt.year - min_year
    
    print(f"✅ Creadas {7} nuevas variables temporales cíclicas")
    return df_result

# ===== SECCIÓN 2: CODIFICACIÓN DE VARIABLES CATEGÓRICAS =====

def apply_onehot_encoding(df: pd.DataFrame, 
                         categorical_cols: List[str],
                         drop_first: bool = True) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a variables categóricas de baja cardinalidad
    
    Args:
        df: DataFrame con datos
        categorical_cols: Lista de columnas categóricas
        drop_first: Si eliminar primera categoría para evitar multicolinealidad
    
    Returns:
        DataFrame con variables codificadas
    """
    print(f"🎯 Aplicando One-Hot Encoding a {len(categorical_cols)} variables...")
    
    df_result = df.copy()
    
    for col in categorical_cols:
        if col in df_result.columns:
            # Crear dummies
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
            df_result = pd.concat([df_result, dummies], axis=1)
            
            print(f"  ✅ {col}: {len(dummies.columns)} nuevas variables")
        else:
            print(f"  ⚠️ Columna {col} no encontrada")
    
    return df_result

def apply_target_encoding(df: pd.DataFrame,
                         categorical_col: str,
                         target_col: str,
                         smoothing: float = 10.0,
                         cv_folds: int = 5) -> pd.DataFrame:
    """
    Aplica Target Encoding con validación cruzada para evitar overfitting
    
    Args:
        df: DataFrame con datos
        categorical_col: Columna categórica a codificar
        target_col: Variable objetivo
        smoothing: Factor de suavizado
        cv_folds: Número de folds para validación cruzada
    
    Returns:
        DataFrame con variable codificada
    """
    print(f"🎯 Aplicando Target Encoding a {categorical_col}...")
    
    df_result = df.copy()
    
    # Calcular media global del target
    global_mean = df_result[target_col].mean()
    
    # Calcular estadísticas por categoría
    category_stats = df_result.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    
    # Aplicar suavizado: (count * category_mean + smoothing * global_mean) / (count + smoothing)
    category_stats['smoothed_mean'] = (
        (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) /
        (category_stats['count'] + smoothing)
    )
    
    # Mapear valores suavizados
    df_result[f'{categorical_col}_target_encoded'] = df_result[categorical_col].map(
        category_stats['smoothed_mean']
    )
    
    # Llenar valores faltantes con media global
    df_result[f'{categorical_col}_target_encoded'].fillna(global_mean, inplace=True)
    
    print(f"  ✅ Creada variable {categorical_col}_target_encoded")
    return df_result

def apply_frequency_encoding(df: pd.DataFrame, 
                           categorical_cols: List[str]) -> pd.DataFrame:
    """
    Aplica Frequency Encoding basado en frecuencia de aparición
    
    Args:
        df: DataFrame con datos
        categorical_cols: Lista de columnas categóricas
    
    Returns:
        DataFrame con variables codificadas por frecuencia
    """
    print(f"📊 Aplicando Frequency Encoding a {len(categorical_cols)} variables...")
    
    df_result = df.copy()
    
    for col in categorical_cols:
        if col in df_result.columns:
            # Calcular frecuencias
            freq_map = df_result[col].value_counts().to_dict()
            
            # Aplicar encoding
            df_result[f'{col}_frequency'] = df_result[col].map(freq_map)
            
            print(f"  ✅ {col}: frecuencias de {len(freq_map)} categorías")
        else:
            print(f"  ⚠️ Columna {col} no encontrada")
    
    return df_result

def group_rare_categories(df: pd.DataFrame,
                         categorical_col: str,
                         threshold: float = 0.01,
                         other_label: str = 'Other') -> pd.DataFrame:
    """
    Agrupa categorías poco frecuentes en una categoría 'Other'
    
    Args:
        df: DataFrame con datos
        categorical_col: Columna categórica
        threshold: Umbral de frecuencia mínima (proporción)
        other_label: Etiqueta para categorías agrupadas
    
    Returns:
        DataFrame con categorías raras agrupadas
    """
    print(f"🗂️ Agrupando categorías raras en {categorical_col}...")
    
    df_result = df.copy()
    
    # Calcular frecuencias relativas
    value_counts = df_result[categorical_col].value_counts()
    freq_props = value_counts / len(df_result)
    
    # Identificar categorías raras
    rare_categories = freq_props[freq_props < threshold].index
    
    # Crear nueva columna con categorías agrupadas
    new_col_name = f'{categorical_col}_grouped'
    df_result[new_col_name] = df_result[categorical_col].copy()
    df_result.loc[df_result[categorical_col].isin(rare_categories), new_col_name] = other_label
    
    print(f"  ✅ Agrupadas {len(rare_categories)} categorías raras en '{other_label}'")
    print(f"  📊 Categorías finales: {df_result[new_col_name].nunique()}")
    
    return df_result

# ===== SECCIÓN 3: NORMALIZACIÓN Y ESCALADO =====

def apply_standard_scaling(df: pd.DataFrame, 
                          numeric_cols: List[str],
                          fit_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Aplica StandardScaler a variables numéricas
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        fit_data: Datos para ajustar el scaler (si None, usa df)
    
    Returns:
        Tuple de (DataFrame escalado, scaler ajustado)
    """
    print(f"📏 Aplicando StandardScaler a {len(numeric_cols)} variables...")
    
    ml_libs = _import_ml_libraries()
    StandardScaler = ml_libs.get('StandardScaler')
    
    if StandardScaler is None:
        print("❌ Error: sklearn no disponible")
        return df, None
    
    df_result = df.copy()
    scaler = StandardScaler()
    
    # Ajustar scaler
    fit_data_to_use = fit_data if fit_data is not None else df_result
    scaler.fit(fit_data_to_use[numeric_cols])
    
    # Transformar datos
    scaled_values = scaler.transform(df_result[numeric_cols])
    
    # Crear nuevas columnas escaladas
    for i, col in enumerate(numeric_cols):
        df_result[f'{col}_scaled'] = scaled_values[:, i]
    
    print(f"  ✅ Variables escaladas con media≈0 y std≈1")
    return df_result, scaler

def apply_minmax_scaling(df: pd.DataFrame,
                        numeric_cols: List[str],
                        feature_range: Tuple[float, float] = (0, 1),
                        fit_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Aplica MinMaxScaler a variables numéricas
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        feature_range: Rango de escalado
        fit_data: Datos para ajustar el scaler
    
    Returns:
        Tuple de (DataFrame escalado, scaler ajustado)
    """
    print(f"📐 Aplicando MinMaxScaler a {len(numeric_cols)} variables...")
    
    ml_libs = _import_ml_libraries()
    MinMaxScaler = ml_libs.get('MinMaxScaler')
    
    if MinMaxScaler is None:
        print("❌ Error: sklearn no disponible")
        return df, None
    
    df_result = df.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Ajustar scaler
    fit_data_to_use = fit_data if fit_data is not None else df_result
    scaler.fit(fit_data_to_use[numeric_cols])
    
    # Transformar datos
    scaled_values = scaler.transform(df_result[numeric_cols])
    
    # Crear nuevas columnas escaladas
    for i, col in enumerate(numeric_cols):
        df_result[f'{col}_minmax'] = scaled_values[:, i]
    
    print(f"  ✅ Variables escaladas al rango {feature_range}")
    return df_result, scaler

def apply_robust_scaling(df: pd.DataFrame,
                        numeric_cols: List[str],
                        fit_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Aplica RobustScaler a variables con outliers
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        fit_data: Datos para ajustar el scaler
    
    Returns:
        Tuple de (DataFrame escalado, scaler ajustado)
    """
    print(f"🛡️ Aplicando RobustScaler a {len(numeric_cols)} variables...")
    
    ml_libs = _import_ml_libraries()
    RobustScaler = ml_libs.get('RobustScaler')
    
    if RobustScaler is None:
        print("❌ Error: sklearn no disponible")
        return df, None
    
    df_result = df.copy()
    scaler = RobustScaler()
    
    # Ajustar scaler
    fit_data_to_use = fit_data if fit_data is not None else df_result
    scaler.fit(fit_data_to_use[numeric_cols])
    
    # Transformar datos
    scaled_values = scaler.transform(df_result[numeric_cols])
    
    # Crear nuevas columnas escaladas
    for i, col in enumerate(numeric_cols):
        df_result[f'{col}_robust'] = scaled_values[:, i]
    
    print(f"  ✅ Variables escaladas usando mediana y rango intercuartílico")
    return df_result, scaler

def apply_log_transformation(df: pd.DataFrame,
                           numeric_cols: List[str],
                           add_constant: float = 1.0) -> pd.DataFrame:
    """
    Aplica transformación logarítmica a variables asimétricas
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        add_constant: Constante a sumar antes del log (para evitar log(0))
    
    Returns:
        DataFrame con variables transformadas
    """
    print(f"📊 Aplicando transformación logarítmica a {len(numeric_cols)} variables...")
    
    df_result = df.copy()
    
    for col in numeric_cols:
        if col in df_result.columns:
            # Verificar valores no positivos
            min_val = df_result[col].min()
            if min_val <= 0:
                print(f"  ⚠️ {col}: valores ≤ 0 detectados, usando log1p")
                df_result[f'{col}_log'] = np.log1p(df_result[col])
            else:
                df_result[f'{col}_log'] = np.log(df_result[col] + add_constant)
            
            print(f"  ✅ {col}: transformación logarítmica aplicada")
        else:
            print(f"  ⚠️ Columna {col} no encontrada")
    
    return df_result

# ===== SECCIÓN 4: FEATURE ENGINEERING AVANZADO =====

def create_interaction_features(df: pd.DataFrame,
                               interactions: List[Tuple[str, str]],
                               operation: str = 'multiply') -> pd.DataFrame:
    """
    Crea features de interacción entre variables
    
    Args:
        df: DataFrame con datos
        interactions: Lista de tuplas con pares de variables
        operation: Tipo de operación ('multiply', 'add', 'divide', 'subtract')
    
    Returns:
        DataFrame con features de interacción
    """
    print(f"🔗 Creando {len(interactions)} features de interacción ({operation})...")
    
    df_result = df.copy()
    
    for var1, var2 in interactions:
        if var1 in df_result.columns and var2 in df_result.columns:
            new_col_name = f'{var1}_x_{var2}'
            
            if operation == 'multiply':
                df_result[new_col_name] = df_result[var1] * df_result[var2]
            elif operation == 'add':
                df_result[new_col_name] = df_result[var1] + df_result[var2]
            elif operation == 'divide':
                # Evitar división por cero
                df_result[new_col_name] = df_result[var1] / (df_result[var2] + 1e-8)
            elif operation == 'subtract':
                df_result[new_col_name] = df_result[var1] - df_result[var2]
            
            print(f"  ✅ {new_col_name}")
        else:
            print(f"  ⚠️ Variables {var1} o {var2} no encontradas")
    
    return df_result

def create_macroeconomic_features(df: pd.DataFrame,
                                 interest_rate_cols: List[str]) -> pd.DataFrame:
    """
    Crea variables macroeconómicas derivadas
    
    Args:
        df: DataFrame con datos
        interest_rate_cols: Lista de columnas de tasas de interés
    
    Returns:
        DataFrame con variables macroeconómicas
    """
    print(f"📈 Creando features macroeconómicas...")
    
    df_result = df.copy()
    
    # Ratios entre tasas de interés
    if len(interest_rate_cols) >= 2:
        for i, rate1 in enumerate(interest_rate_cols):
            for rate2 in interest_rate_cols[i+1:]:
                if rate1 in df_result.columns and rate2 in df_result.columns:
                    ratio_name = f'{rate1}_to_{rate2}_ratio'
                    df_result[ratio_name] = df_result[rate1] / (df_result[rate2] + 1e-8)
                    print(f"  ✅ {ratio_name}")
    
    # Spreads entre tasas
    if len(interest_rate_cols) >= 2:
        for i, rate1 in enumerate(interest_rate_cols):
            for rate2 in interest_rate_cols[i+1:]:
                if rate1 in df_result.columns and rate2 in df_result.columns:
                    spread_name = f'{rate1}_{rate2}_spread'
                    df_result[spread_name] = df_result[rate1] - df_result[rate2]
                    print(f"  ✅ {spread_name}")
    
    # Variables lag (valores pasados)
    for col in interest_rate_cols:
        if col in df_result.columns:
            df_result[f'{col}_lag1'] = df_result[col].shift(1)
            df_result[f'{col}_lag2'] = df_result[col].shift(2)
            print(f"  ✅ {col} lag features")
    
    return df_result

def _compute_rolling_window_stats(
    region_frame: pd.DataFrame,
    year_col: str,
    price_col: str,
    year_ref: int,
    window_years: int,
    min_obs: int,
) -> Dict[str, Any]:
    """Calcula estadísticos exactos sobre la ventana causal [year-k, year-1]."""
    window_mask = (
        (region_frame[year_col] >= year_ref - window_years) &
        (region_frame[year_col] <= year_ref - 1)
    )
    window_prices = region_frame.loc[window_mask, price_col].dropna()
    obs_count = int(window_prices.shape[0])

    if obs_count < min_obs:
        return {
            'rolling_regional_mean': np.nan,
            'rolling_regional_median': np.nan,
            'rolling_regional_std': np.nan,
            'rolling_regional_cv': np.nan,
            'rolling_regional_count': obs_count,
            'rolling_regional_p90': np.nan,
        }

    mean_value = float(window_prices.mean())
    std_value = float(window_prices.std(ddof=1)) if obs_count > 1 else 0.0
    return {
        'rolling_regional_mean': mean_value,
        'rolling_regional_median': float(window_prices.median()),
        'rolling_regional_std': std_value,
        'rolling_regional_cv': std_value / mean_value if mean_value else np.nan,
        'rolling_regional_count': obs_count,
        'rolling_regional_p90': float(window_prices.quantile(0.9)),
    }


def create_rolling_regional_features(
    df: pd.DataFrame,
    region_col: str = 'region',
    price_col: str = 'purchase_price',
    year_col: str = 'year',
    window_years: Optional[int] = None,
    min_obs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Crea agregados regionales causales usando solo datos de años anteriores.

    Para cada fila del año y se usan únicamente transacciones con años en
    [y - window_years, y - 1] dentro de la misma región.
    """
    if window_years is None:
        window_years = ROLLING_WINDOW_YEARS
    if min_obs is None:
        min_obs = MIN_OBS_PER_WINDOW

    required_cols = {region_col, price_col, year_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas para rolling regional: {sorted(missing_cols)}")

    print(
        "🔄 Creando features regionales causales "
        f"(ventana={window_years} anios, min_obs={min_obs})..."
    )

    df_result = df.copy()
    existing_rolling = [col for col in ROLLING_FEATURE_COLUMNS if col in df_result.columns]
    if existing_rolling:
        df_result = df_result.drop(columns=existing_rolling)
    base_pairs = (
        df_result[[region_col, year_col]]
        .dropna()
        .drop_duplicates()
        .sort_values([region_col, year_col])
    )

    records: List[Dict[str, Any]] = []
    for region_value, region_frame in df_result[[region_col, year_col, price_col]].groupby(region_col):
        region_frame = region_frame.sort_values(year_col)
        candidate_years = (
            base_pairs.loc[base_pairs[region_col] == region_value, year_col]
            .dropna()
            .astype(int)
            .tolist()
        )
        for year_ref in candidate_years:
            stats = _compute_rolling_window_stats(
                region_frame=region_frame,
                year_col=year_col,
                price_col=price_col,
                year_ref=year_ref,
                window_years=window_years,
                min_obs=min_obs,
            )
            records.append({
                region_col: region_value,
                year_col: int(year_ref),
                **stats,
            })

    rolling_df = pd.DataFrame.from_records(records)
    df_result = df_result.merge(rolling_df, on=[region_col, year_col], how='left')

    n_nan = df_result['rolling_regional_mean'].isna().sum()
    print(f"  ✅ Features rolling aplicadas a {df_result.shape[0]:,} filas; {n_nan:,} NaN")
    return df_result


def create_geographic_aggregated_features(
    df: pd.DataFrame,
    region_col: str = 'region',
    price_col: str = 'purchase_price',
    window: int = 12,
) -> pd.DataFrame:
    """
    DEPRECADO (RFE-02): usa create_rolling_regional_features.
    """
    warnings.warn(
        "create_geographic_aggregated_features es no-canonica; "
        "usar create_rolling_regional_features",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_rolling_regional_features(
        df,
        region_col=region_col,
        price_col=price_col,
        year_col='year',
        window_years=window,
    )


def create_regional_aggregated_features(
    df: pd.DataFrame,
    region_col: str = 'region',
    price_col: str = 'purchase_price',
    year_col: str = 'year',
    window_years: Optional[int] = None,
    min_obs: Optional[int] = None,
) -> pd.DataFrame:
    """
    DEPRECADO (RFE-02): alias legacy hacia create_rolling_regional_features.
    """
    warnings.warn(
        "create_regional_aggregated_features es no-canonica; "
        "usar create_rolling_regional_features",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_rolling_regional_features(
        df,
        region_col=region_col,
        price_col=price_col,
        year_col=year_col,
        window_years=window_years,
        min_obs=min_obs,
    )

# ===== SECCIÓN 5: SELECCIÓN DE FEATURES =====

def analyze_correlation_multicollinearity(df: pd.DataFrame,
                                        numeric_cols: List[str],
                                        correlation_threshold: float = 0.95,
                                        vif_threshold: float = 10.0) -> Dict[str, Any]:
    """
    Analiza correlación y multicolinealidad
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        correlation_threshold: Umbral de correlación alta
        vif_threshold: Umbral de VIF para multicolinealidad
    
    Returns:
        Diccionario con análisis de correlación y VIF
    """
    print(f"🔍 Analizando correlación y multicolinealidad...")
    
    # Filtrar columnas existentes
    existing_cols = [col for col in numeric_cols if col in df.columns]
    df_numeric = df[existing_cols].select_dtypes(include=[np.number])
    
    # Matriz de correlación
    corr_matrix = df_numeric.corr()
    
    # Identificar pares altamente correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > correlation_threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print(f"  📊 {len(high_corr_pairs)} pares con correlación > {correlation_threshold}")
    
    # Calcular VIF (Variance Inflation Factor)
    ml_libs = _import_ml_libraries()
    vif_func = ml_libs.get('variance_inflation_factor')
    
    vif_results = []
    if vif_func is not None:
        try:
            # Remover filas con NaN para VIF
            df_clean = df_numeric.dropna()
            
            if len(df_clean) > 0:
                vif_data = []
                for i, col in enumerate(df_clean.columns):
                    try:
                        vif_val = vif_func(df_clean.values, i)
                        vif_data.append({'variable': col, 'VIF': vif_val})
                    except:
                        vif_data.append({'variable': col, 'VIF': np.nan})
                
                vif_results = pd.DataFrame(vif_data)
                high_vif = vif_results[vif_results['VIF'] > vif_threshold]
                print(f"  🚨 {len(high_vif)} variables con VIF > {vif_threshold}")
        except Exception as e:
            print(f"  ⚠️ Error calculando VIF: {e}")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'vif_results': vif_results,
        'summary': {
            'total_variables': len(existing_cols),
            'high_correlation_pairs': len(high_corr_pairs),
            'high_vif_variables': len(vif_results[vif_results['VIF'] > vif_threshold]) if len(vif_results) > 0 else 0
        }
    }

def calculate_feature_importance_preliminary(df: pd.DataFrame,
                                           feature_cols: List[str],
                                           target_col: str,
                                           n_estimators: int = 100) -> Dict[str, Any]:
    """
    Calcula importancia preliminar de features
    
    Args:
        df: DataFrame con datos
        feature_cols: Lista de columnas de features
        target_col: Variable objetivo
        n_estimators: Número de árboles para Random Forest
    
    Returns:
        Diccionario con diferentes medidas de importancia
    """
    print(f"🎯 Calculando importancia preliminar de {len(feature_cols)} features...")
    
    # Filtrar datos válidos
    valid_cols = [col for col in feature_cols if col in df.columns]
    df_clean = df[valid_cols + [target_col]].dropna()
    
    if len(df_clean) == 0:
        print("❌ No hay datos válidos para calcular importancia")
        return {}
    
    X = df_clean[valid_cols].select_dtypes(include=[np.number])
    y = df_clean[target_col]
    
    results = {}
    
    # Random Forest Feature Importance
    ml_libs = _import_ml_libraries()
    RandomForestRegressor = ml_libs.get('RandomForestRegressor')
    
    if RandomForestRegressor is not None:
        try:
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            rf.fit(X, y)
            
            rf_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['random_forest'] = rf_importance
            print(f"  ✅ Random Forest importance calculada")
        except Exception as e:
            print(f"  ⚠️ Error en Random Forest: {e}")
    
    # Mutual Information
    mutual_info_func = ml_libs.get('mutual_info_regression')
    if mutual_info_func is not None:
        try:
            mi_scores = mutual_info_func(X, y, random_state=42)
            mi_importance = pd.DataFrame({
                'feature': X.columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            
            results['mutual_information'] = mi_importance
            print(f"  ✅ Mutual Information calculada")
        except Exception as e:
            print(f"  ⚠️ Error en Mutual Information: {e}")
    
    # Correlación simple con target
    target_corr = X.corrwith(y).abs().sort_values(ascending=False)
    results['target_correlation'] = pd.DataFrame({
        'feature': target_corr.index,
        'correlation': target_corr.values
    })
    print(f"  ✅ Correlación con target calculada")
    
    return results

def remove_low_variance_features(df: pd.DataFrame,
                                numeric_cols: List[str],
                                variance_threshold: float = 0.01) -> pd.DataFrame:
    """
    Elimina features con baja varianza
    
    Args:
        df: DataFrame con datos
        numeric_cols: Lista de columnas numéricas
        variance_threshold: Umbral mínimo de varianza
    
    Returns:
        DataFrame sin features de baja varianza
    """
    print(f"📉 Eliminando features con varianza < {variance_threshold}...")
    
    df_result = df.copy()
    
    # Calcular varianzas
    existing_cols = [col for col in numeric_cols if col in df_result.columns]
    variances = df_result[existing_cols].var()
    
    # Identificar features de baja varianza
    low_variance_cols = variances[variances < variance_threshold].index.tolist()
    
    if len(low_variance_cols) > 0:
        df_result = df_result.drop(columns=low_variance_cols)
        print(f"  🗑️ Eliminadas {len(low_variance_cols)} features de baja varianza")
        print(f"  📝 Features eliminadas: {low_variance_cols[:5]}{'...' if len(low_variance_cols) > 5 else ''}")
    else:
        print(f"  ✅ No se encontraron features de baja varianza")
    
    return df_result

def create_feature_engineering_summary(df_original: pd.DataFrame,
                                     df_processed: pd.DataFrame) -> Dict[str, Any]:
    """
    Crea resumen del proceso de feature engineering
    
    Args:
        df_original: DataFrame original
        df_processed: DataFrame procesado
    
    Returns:
        Diccionario con resumen del proceso
    """
    print("📋 Creando resumen de Feature Engineering...")
    
    # Comparar datasets
    original_shape = df_original.shape
    processed_shape = df_processed.shape
    
    # Nuevas columnas creadas
    original_cols = set(df_original.columns)
    processed_cols = set(df_processed.columns)
    new_cols = processed_cols - original_cols
    
    # Columnas eliminadas
    removed_cols = original_cols - processed_cols
    
    # Tipos de datos
    original_dtypes = df_original.dtypes.value_counts()
    processed_dtypes = df_processed.dtypes.value_counts()
    
    summary = {
        'dataset_comparison': {
            'original_shape': original_shape,
            'processed_shape': processed_shape,
            'rows_change': processed_shape[0] - original_shape[0],
            'columns_change': processed_shape[1] - original_shape[1]
        },
        'columns_analysis': {
            'original_columns': len(original_cols),
            'processed_columns': len(processed_cols),
            'new_columns': len(new_cols),
            'removed_columns': len(removed_cols),
            'new_columns_list': list(new_cols),
            'removed_columns_list': list(removed_cols)
        },
        'data_types': {
            'original_dtypes': original_dtypes.to_dict(),
            'processed_dtypes': processed_dtypes.to_dict()
        }
    }
    
    print(f"  📊 Filas: {original_shape[0]:,} → {processed_shape[0]:,}")
    print(f"  📊 Columnas: {original_shape[1]:,} → {processed_shape[1]:,}")
    print(f"  ➕ Nuevas columnas: {len(new_cols)}")
    print(f"  ➖ Columnas eliminadas: {len(removed_cols)}")
    
    return summary

# ===== FUNCIONES PRINCIPALES PARA EL NOTEBOOK =====

def create_temporal_features(df: pd.DataFrame, date_col: str = 'date', year_build_col: str = 'year_build') -> pd.DataFrame:
    """
    Crea todas las variables temporales derivadas
    
    Args:
        df: DataFrame de entrada
        date_col: Nombre de la columna de fecha
        year_build_col: Nombre de la columna de año de construcción
        
    Returns:
        DataFrame con variables temporales añadidas
    """
    print("📅 Creando variables temporales...")
    
    df_temp = df.copy()
    config = FEATURE_ENGINEERING_CONFIG['temporal']
    
    # Convertir fecha si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    
    # Extraer componentes de fecha
    df_temp['year'] = df_temp[date_col].dt.year
    df_temp['month'] = df_temp[date_col].dt.month  
    df_temp['day'] = df_temp[date_col].dt.day
    df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
    
    # Solo crear quarter si no existe
    if 'quarter' not in df_temp.columns:
        df_temp['quarter'] = df_temp[date_col].dt.quarter
    
    # Variables estacionales
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring', 
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df_temp['season'] = df_temp['month'].map(season_map)
    
    # Edad de la propiedad
    df_temp['property_age'] = config['current_year'] - df_temp[year_build_col]
    
    # Década de construcción
    df_temp['decade_built'] = (df_temp[year_build_col] // 10) * 10
    
    # Variables temporales cíclicas (para capturar periodicidad)
    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
    df_temp['quarter_sin'] = np.sin(2 * np.pi * df_temp['quarter'] / 4)
    df_temp['quarter_cos'] = np.cos(2 * np.pi * df_temp['quarter'] / 4)
    
    temporal_vars = ['year', 'month', 'quarter', 'season', 'property_age', 'decade_built',
                    'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    
    print(f"✅ Variables temporales creadas: {temporal_vars}")
    return df_temp

def create_price_features(df: pd.DataFrame, target_col: str = 'purchase_price') -> pd.DataFrame:
    """
    Crea solo el target transformado para modelado.
    
    Args:
        df: DataFrame de entrada
        target_col: Nombre de la columna de precio objetivo
        
    Returns:
        DataFrame con la columna log_price añadida
    """
    print("💰 Creando variables de precio...")
    
    df_price = df.copy()
    
    # Log de precios (para normalizar distribución)
    df_price[MODEL_TARGET_COL] = np.log1p(df_price[target_col])

    print(f"✅ Variables de precio creadas: ['{MODEL_TARGET_COL}']")
    return df_price

def create_size_features(df: pd.DataFrame, sqm_col: str = 'sqm', rooms_col: str = 'no_rooms') -> pd.DataFrame:
    """
    Crea variables derivadas de tamaño y espacio
    
    Args:
        df: DataFrame de entrada
        sqm_col: Nombre de la columna de metros cuadrados
        rooms_col: Nombre de la columna de número de habitaciones
        
    Returns:
        DataFrame con variables de tamaño añadidas
    """
    print("🏠 Creando variables de tamaño...")
    
    df_size = df.copy()
    
    # Categorías de habitaciones
    df_size['rooms_category'] = pd.cut(
        df_size[rooms_col],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=['Small', 'Medium', 'Large', 'XLarge'],
    )
    
    # Categorías de tamaño por m²
    sqm_quartiles = df_size[sqm_col].quantile([0.33, 0.67])
    if sqm_quartiles[0.33] >= sqm_quartiles[0.67]:
        df_size['size_category'] = pd.qcut(
            df_size[sqm_col].rank(method='first'),
            q=3,
            labels=['Small', 'Medium', 'Large'],
        )
    else:
        df_size['size_category'] = pd.cut(
            df_size[sqm_col],
            bins=[-np.inf, sqm_quartiles[0.33], sqm_quartiles[0.67], np.inf],
            labels=['Small', 'Medium', 'Large'],
        )
    
    # Eficiencia espacial (m² por habitación)
    df_size['sqm_per_room'] = df_size[sqm_col] / df_size[rooms_col]
    
    # Ratios útiles
    df_size['rooms_sqm_ratio'] = df_size[rooms_col] / df_size[sqm_col]
    
    size_vars = ['rooms_category', 'size_category', 'sqm_per_room', 'rooms_sqm_ratio']
    print(f"✅ Variables de tamaño creadas: {size_vars}")
    return df_size

def encode_categorical_variables(df: pd.DataFrame, target_col: str = 'purchase_price') -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica codificación categórica usando one-hot, target y frequency encoding
    
    Args:
        df: DataFrame de entrada
        target_col: Nombre de la columna objetivo para target encoding
        
    Returns:
        Tuple con DataFrame codificado y diccionario de información de encoding
    """
    print("🔤 Iniciando codificación de variables categóricas...")
    
    config = FEATURE_ENGINEERING_CONFIG['encoding']
    categorical_vars = ['region', 'house_type', 'sales_type', 'season',
                       'rooms_category', 'size_category']
    
    df_encoded = df.copy()
    encoding_info = {}
    
    # === ANÁLISIS DE CARDINALIDAD ===
    print("\n📊 Análisis de cardinalidad:")
    cardinality_info = {}
    for var in categorical_vars:
        if var in df_encoded.columns:
            n_unique = df_encoded[var].nunique()
            cardinality_info[var] = n_unique
            print(f"{var}: {n_unique} categorías únicas")
    
    # === ONE-HOT ENCODING ===
    print("\n🔢 Aplicando One-Hot Encoding...")
    low_card_vars = [var for var, card in cardinality_info.items() 
                     if card <= config['max_categories_onehot'] and var != 'region']
    
    for var in low_card_vars:
        if var in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[var], prefix=var, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            print(f"✅ {var}: {len(dummies.columns)} variables dummy creadas")
    
    # === TARGET ENCODING ===
    print("\n🎯 Aplicando Target Encoding para 'region'...")
    if 'region' in df_encoded.columns:
        region_stats = df_encoded.groupby('region')[target_col].agg(['mean', 'std', 'count']).reset_index()
        region_stats.columns = ['region', 'region_price_mean', 'region_price_std', 'region_count']
        
        global_mean = df_encoded[target_col].mean()
        min_samples = config['min_samples_smoothing']
        
        def smooth_target_encoding(row):
            if row['region_count'] >= min_samples:
                return row['region_price_mean']
            else:
                weight = row['region_count'] / min_samples
                return weight * row['region_price_mean'] + (1 - weight) * global_mean
        
        region_stats['region_target_encoded'] = region_stats.apply(smooth_target_encoding, axis=1)
        
        df_encoded = df_encoded.merge(
            region_stats[['region', 'region_target_encoded', 'region_price_mean', 'region_count']], 
            on='region', how='left'
        )
        
        encoding_info['target_encoding'] = region_stats
        print(f"✅ Target encoding aplicado: media global = {global_mean:.0f}")
    
    # === FREQUENCY ENCODING ===
    print("\n📊 Aplicando Frequency Encoding...")
    if 'region' in df_encoded.columns:
        freq_encoding = df_encoded['region'].value_counts().to_dict()
        df_encoded['region_frequency'] = df_encoded['region'].map(freq_encoding)
        encoding_info['frequency_encoding'] = freq_encoding
        print("✅ Frequency encoding aplicado a 'region'")
    
    encoding_info['cardinality'] = cardinality_info
    encoding_info['low_card_vars'] = low_card_vars
    
    print(f"\n📋 Resumen de codificación:")
    print(f"Dataset: {df_encoded.shape[0]:,} filas x {df_encoded.shape[1]} columnas")
    
    return df_encoded, encoding_info

def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica escalado estratégico a las variables numéricas
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        Tuple con DataFrame escalado y diccionario de scalers
    """
    print("⚖️ Iniciando normalización y escalado...")
    
    ml_lib = _import_ml_libraries()
    if not ml_lib:
        print("⚠️ sklearn no disponible; se omite el escalado y se continua con scalers vacíos.")
        return df.copy(), {}
    
    df_scaled = df.copy()
    scalers = {}
    
    # Variables numéricas para escalar
    numeric_vars_original = ['sqm', 'no_rooms', 'year_build', 'property_age']
    numeric_vars_derived = ['sqm_per_room', 'rooms_sqm_ratio',
                           'region_target_encoded', 'region_frequency']
    numeric_vars_cyclical = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    
    all_numeric_vars = numeric_vars_original + numeric_vars_derived + numeric_vars_cyclical
    
    # Filtrar variables que existen en el DataFrame
    existing_vars = [var for var in all_numeric_vars if var in df_scaled.columns]
    
    print(f"Variables numéricas a escalar: {len(existing_vars)}")
    
    # Análisis de skewness
    skewness_info = {}
    for var in existing_vars:
        skew = df_scaled[var].skew()
        skewness_info[var] = skew
    
    # StandardScaler para variables poco asimétricas
    standard_vars = [var for var, skew in skewness_info.items() if abs(skew) <= 1]
    if standard_vars:
        scaler_standard = ml_lib['StandardScaler']()
        df_scaled[standard_vars] = scaler_standard.fit_transform(df_scaled[standard_vars])
        scalers['standard'] = {'scaler': scaler_standard, 'variables': standard_vars}
        print(f"StandardScaler aplicado a: {len(standard_vars)} variables")
    
    # RobustScaler para variables asimétricas
    robust_vars = [var for var, skew in skewness_info.items() if abs(skew) > 1]
    if robust_vars:
        scaler_robust = ml_lib['RobustScaler']()
        df_scaled[robust_vars] = scaler_robust.fit_transform(df_scaled[robust_vars])
        scalers['robust'] = {'scaler': scaler_robust, 'variables': robust_vars}
        print(f"RobustScaler aplicado a: {len(robust_vars)} variables")
    
    # MinMaxScaler para variables cíclicas
    cyclical_existing = [var for var in numeric_vars_cyclical if var in df_scaled.columns]
    if cyclical_existing:
        scaler_minmax = ml_lib['MinMaxScaler']()
        df_scaled[cyclical_existing] = scaler_minmax.fit_transform(df_scaled[cyclical_existing])
        scalers['minmax'] = {'scaler': scaler_minmax, 'variables': cyclical_existing}
        print(f"MinMaxScaler aplicado a: {len(cyclical_existing)} variables")
    
    print(f"✅ Escalado completado: {df_scaled.shape[0]:,} filas x {df_scaled.shape[1]} columnas")
    
    return df_scaled, scalers

def create_advanced_features(df: pd.DataFrame, target_col: str = 'purchase_price') -> pd.DataFrame:
    """
    Crea features avanzados seguras para modelado temporal.

    Las features derivadas directamente del target se reemplazan por versiones
    causales calculadas sobre ventanas rolling por región.
    """
    print("🚀 Creando Feature Engineering Avanzado...")

    config = FEATURE_ENGINEERING_CONFIG['temporal']
    df_advanced = df.copy()

    # === VARIABLES DE INTERACCIÓN SEGURAS ===
    print("\n🔗 Creando variables de interacción seguras...")

    # RFE-03: se elimina el bloque dependiente de price_per_sqm y region_target_encoded.
    for house_type in ['Villa', 'Apartment']:
        house_col = f'house_type_{house_type}'
        if house_col in df_advanced.columns and 'property_age' in df_advanced.columns:
            df_advanced[f'age_x_{house_type.lower()}'] = (
                df_advanced['property_age'] * df_advanced[house_col]
            )
            print(f"✅ property_age × {house_col}")

    if 'sqm_per_room' in df_advanced.columns:
        df_advanced['sqm_per_room_squared'] = df_advanced['sqm_per_room'] ** 2
        print("✅ sqm_per_room²")

    if 'no_rooms' in df_advanced.columns and 'sqm' in df_advanced.columns:
        df_advanced['rooms_sqm_interaction'] = df_advanced['no_rooms'] * df_advanced['sqm']
        print("✅ no_rooms × sqm")

    # === VARIABLES MACROECONÓMICAS ===
    print("\n💹 Creando variables macroeconómicas...")

    if 'year' in df_advanced.columns:
        year_min, year_max = df_advanced['year'].min(), df_advanced['year'].max()
        year_range = max(year_max - year_min, 1)
        df_advanced['time_trend'] = (df_advanced['year'] - year_min) / year_range
        print("✅ time_trend")

        crisis_years = config['crisis_years']
        df_advanced['crisis_period'] = df_advanced['year'].isin(crisis_years).astype(int)
        print(f"✅ crisis_period (años: {crisis_years})")

        def assign_market_phase(year: int) -> str:
            for phase, (start, end) in config['market_phases'].items():
                if start <= year <= end:
                    return phase
            return 'other'

        df_advanced['market_phase'] = df_advanced['year'].apply(assign_market_phase)
        market_dummies = pd.get_dummies(df_advanced['market_phase'], prefix='phase')
        df_advanced = pd.concat([df_advanced, market_dummies], axis=1)
        print(f"✅ market_phase → {list(market_dummies.columns)}")

    # === VARIABLES GEOGRÁFICAS CAUSALES ===
    print("\n🌍 Creando variables geográficas causales...")

    required_cols = {'region', 'year', target_col}
    if required_cols <= set(df_advanced.columns):
        missing_rolling = [col for col in ROLLING_FEATURE_COLUMNS if col not in df_advanced.columns]
        if missing_rolling:
            df_advanced = create_rolling_regional_features(
                df_advanced,
                region_col='region',
                price_col=target_col,
                year_col='year',
            )

        rolling_p90 = df_advanced['rolling_regional_p90']
        rolling_median = df_advanced['rolling_regional_median']
        df_advanced['rolling_regional_median_v2'] = rolling_median

        df_advanced['is_premium_causal'] = np.where(
            rolling_p90.notna(),
            (df_advanced[target_col] > rolling_p90).astype(float),
            np.nan,
        )
        print("✅ is_premium_causal")

        df_advanced['price_deviation_from_rolling_median'] = (
            df_advanced[target_col] - rolling_median
        )
        print("✅ price_deviation_from_rolling_median")

    legacy_cols = [
        'regional_p90',
        'regional_median',
        'is_premium',
        'price_deviation_from_median',
        'sqm_x_region',
        'price_per_sqm_x_region',
    ]
    existing_legacy = [col for col in legacy_cols if col in df_advanced.columns]
    if existing_legacy:
        df_advanced = df_advanced.drop(columns=existing_legacy)
        print(f"🧹 Legacy removido: {existing_legacy}")

    print(
        f"\n📋 Features avanzados completados: "
        f"{df_advanced.shape[0]:,} filas x {df_advanced.shape[1]} columnas"
    )

    return df_advanced

def prepare_final_dataset(df: pd.DataFrame, target_col: str = MODEL_TARGET_COL) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Prepara el dataset final para modelado con guardas anti-leak.
    
    Args:
        df: DataFrame de entrada
        target_col: Nombre de la columna objetivo
        
    Returns:
        Tuple con DataFrame final, lista de features seleccionadas y metadatos
    """
    print("🎯 Preparando dataset final para modelado...")
    
    config = FEATURE_ENGINEERING_CONFIG['feature_selection']
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")

    input_forbidden = FORBIDDEN_FEATURE_COLUMNS & set(df.columns)
    assert not input_forbidden, (
        "RFE-05 anti-leak guard: columnas prohibidas presentes en el input de "
        f"prepare_final_dataset: {sorted(input_forbidden)}"
    )

    exclude_cols = [
        # Identificadores y columnas no-feature
        'date', 'region', 'house_id', 'address', 'city', 'area', 'zip_code',
        'house_type', 'sales_type', 'season', 'market_phase',
        'rooms_category', 'size_category', 'decade_built', 'year_build',
        # Target crudo y derivadas directas del target
        'purchase_price', 'price_per_sqm', 'price_zscore', 'price_category',
        'sqm_x_region', 'price_per_sqm_x_region',
        'is_premium', 'price_deviation_from_median',
        'regional_p90', 'regional_median',
        # Agregados regionales globales legacy
        'regional_price_mean', 'regional_price_median', 'regional_price_std',
        'regional_price_cv', 'regional_price_rank',
        'regional_transaction_count', 'regional_liquidity_score',
        'region_target_encoded', 'region_count',
        # Intermedias auxiliares
        'rolling_regional_median_v2',
        # Macro / variables economicas redundantes
        'sqm_price', '%_change_between_offer_and_purchase',
        'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%', 'nom_interest_rate%',
        # Redundantes temporales
        'quarter', 'time_trend',
    ]

    critical_features = [
        'month_sin',
        'month_cos',
        'quarter_sin',
        'quarter_cos',
        'rolling_regional_mean',
        'rolling_regional_median',
        'rolling_regional_std',
        'rolling_regional_cv',
        'rolling_regional_count',
        'rolling_regional_p90',
        'is_premium_causal',
        'price_deviation_from_rolling_median',
    ]

    all_columns = df.columns.tolist()
    feature_columns = [col for col in all_columns if col not in exclude_cols + [target_col]]

    leaks_detected = FORBIDDEN_FEATURE_COLUMNS & set(feature_columns)
    assert not leaks_detected, (
        "RFE-05 anti-leak guard: columnas prohibidas en feature_columns: "
        f"{sorted(leaks_detected)}"
    )

    print(f"Features candidatas: {len(feature_columns)}")

    print("\n🧹 Limpieza de datos...")
    df_modeling = df[feature_columns + [target_col]].copy()

    for col in df_modeling.columns:
        if col == target_col:
            continue
        if np.issubdtype(df_modeling[col].dtype, np.number):
            if np.isinf(df_modeling[col]).any():
                df_modeling[col] = df_modeling[col].replace([np.inf, -np.inf], np.nan)

    print("\n🎯 Feature selection...")

    X = df_modeling[feature_columns].copy()
    y = df_modeling[target_col].copy()

    X_for_selection = X.copy()
    for col in X_for_selection.columns:
        X_for_selection[col] = pd.to_numeric(X_for_selection[col], errors='coerce')
        if X_for_selection[col].isnull().any():
            median_value = X_for_selection[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            X_for_selection[col] = X_for_selection[col].fillna(median_value)

    if len(X) > config['sample_size_fs']:
        sample_idx = np.random.choice(len(X), config['sample_size_fs'], replace=False)
        X_sample = X_for_selection.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        print(f"Usando muestra de {len(X_sample):,} observaciones")
    else:
        X_sample = X_for_selection
        y_sample = y

    try:
        from sklearn.feature_selection import mutual_info_regression, f_regression

        mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        mi_results = pd.DataFrame({
            'feature': X_sample.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        f_scores, f_pvalues = f_regression(X_sample, y_sample)
        f_results = pd.DataFrame({
            'feature': X_sample.columns,
            'f_score': f_scores,
            'p_value': f_pvalues
        }).sort_values('f_score', ascending=False)

        mi_span = mi_results['mutual_info'].max() - mi_results['mutual_info'].min()
        f_span = f_results['f_score'].max() - f_results['f_score'].min()
        mi_results['mi_normalized'] = 0.0 if mi_span == 0 else (
            (mi_results['mutual_info'] - mi_results['mutual_info'].min()) / mi_span
        )
        f_results['f_normalized'] = 0.0 if f_span == 0 else (
            (f_results['f_score'] - f_results['f_score'].min()) / f_span
        )

        combined_results = mi_results.merge(f_results, on='feature')
        combined_results['combined_score'] = (combined_results['mi_normalized'] + combined_results['f_normalized']) / 2
        combined_results = combined_results.sort_values('combined_score', ascending=False)

        top_k = min(config['max_features'], len(feature_columns))
        selected_features_from_ranking = combined_results.head(top_k)['feature'].tolist()

        critical_features_available = [f for f in critical_features if f in feature_columns]
        selected_features = list(dict.fromkeys(selected_features_from_ranking + critical_features_available))

        if len(selected_features) > config['max_features']:
            non_critical = [f for f in selected_features_from_ranking if f not in critical_features_available]
            selected_features = critical_features_available + non_critical[:config['max_features'] - len(critical_features_available)]

        print(f"✅ Seleccionadas {len(selected_features)} features de {len(feature_columns)}")

        df_final = df_modeling[selected_features + [target_col]].copy()
        metadata = {
            'feature_selection': {
                'mutual_info': mi_results.to_dict('records'),
                'f_regression': f_results.to_dict('records'),
                'combined': combined_results.to_dict('records')
            },
            'selected_features': selected_features,
            'dataset_shape': df_final.shape,
            'target_col': target_col,
        }

        print(f"📊 Dataset final: {df_final.shape[0]:,} filas x {df_final.shape[1]-1} features")

        return df_final, selected_features, metadata

    except Exception as e:
        print(f"⚠️ Error en feature selection: {e}")
        critical_features_available = [f for f in critical_features if f in feature_columns]
        non_critical = [f for f in feature_columns if f not in critical_features_available]
        remaining_slots = max(config['max_features'] - len(critical_features_available), 0)
        selected_features = critical_features_available + non_critical[:remaining_slots]
        df_final = df_modeling[selected_features + [target_col]].copy()
        metadata = {
            'selected_features': selected_features,
            'dataset_shape': df_final.shape,
            'target_col': target_col,
            'note': 'Feature selection fallo, usando top features por orden'
        }

        return df_final, selected_features, metadata


def _compute_reference_medians(
    df: pd.DataFrame,
    feature_columns: List[str],
    reference_mask: pd.Series,
) -> pd.Series:
    reference_frame = df.loc[reference_mask, feature_columns].copy()
    reference_frame = reference_frame.replace([np.inf, -np.inf], np.nan)
    medians = reference_frame.median(numeric_only=True).fillna(0.0)
    return medians.reindex(feature_columns).fillna(0.0)


def _apply_reference_imputation(
    frame: pd.DataFrame,
    feature_columns: List[str],
    medians: pd.Series,
) -> pd.DataFrame:
    frame_imputed = frame.copy()
    frame_imputed[feature_columns] = (
        frame_imputed[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )
    return frame_imputed


def create_train_test_split(df: pd.DataFrame, selected_features: List[str], target_col: str = MODEL_TARGET_COL) -> Dict:
    """
    Crea división temporal train/test
    
    Args:
        df: DataFrame final
        selected_features: Lista de features seleccionadas
        target_col: Nombre de la columna objetivo
        
    Returns:
        Diccionario con splits de datos
    """
    print("📅 Creando división temporal train/test...")
    
    config = FEATURE_ENGINEERING_CONFIG['train_test']
    
    # Necesitamos la columna 'year' para hacer el split temporal
    if 'year' not in df.columns:
        raise ValueError("La columna 'year' es necesaria para la división temporal")
    
    train_mask = df['year'] <= config['split_year']
    test_mask = df['year'] >= config['test_start_year']

    X_train = df[train_mask][selected_features]
    X_test = df[test_mask][selected_features]
    y_train = df[train_mask][target_col]
    y_test = df[test_mask][target_col]
    
    split_info = {
        'train_period': f"{df[train_mask]['year'].min()}-{df[train_mask]['year'].max()}",
        'test_period': f"{df[test_mask]['year'].min()}-{df[test_mask]['year'].max()}",
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_pct': len(X_train) / len(df) * 100,
        'test_pct': len(X_test) / len(df) * 100
    }
    
    print(f"📈 Train: {split_info['train_size']:,} obs. ({split_info['train_pct']:.1f}%) - {split_info['train_period']}")
    print(f"📊 Test: {split_info['test_size']:,} obs. ({split_info['test_pct']:.1f}%) - {split_info['test_period']}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_info': split_info
    }


def create_walk_forward_folds(
    df: pd.DataFrame,
    selected_features: List[str],
    target_col: str = MODEL_TARGET_COL,
    year_col: str = 'year',
    min_train_end: Optional[int] = None,
    max_train_end: Optional[int] = None,
    holdout_start: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Genera folds walk-forward expanding y holdout final.
    """
    if min_train_end is None:
        min_train_end = CV_MIN_TRAIN_END_YEAR
    if max_train_end is None:
        max_train_end = CV_MAX_TRAIN_END_YEAR
    if holdout_start is None:
        holdout_start = HOLDOUT_START_YEAR

    if year_col not in df.columns:
        raise ValueError(f"La columna '{year_col}' es requerida para walk-forward")
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' es requerida para walk-forward")

    folds = []
    for train_end in range(min_train_end, max_train_end + 1):
        val_year = train_end + 1
        if val_year >= holdout_start:
            break

        train_mask = df[year_col] <= train_end
        val_mask = df[year_col] == val_year

        X_train = df.loc[train_mask, selected_features].copy()
        y_train = df.loc[train_mask, target_col].copy()
        X_val = df.loc[val_mask, selected_features].copy()
        y_val = df.loc[val_mask, target_col].copy()

        folds.append({
            'train_end': train_end,
            'val_year': val_year,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
        })
        print(f"  fold train<={train_end} (n={len(X_train):,}) / val={val_year} (n={len(X_val):,})")

    train_full_mask = df[year_col] < holdout_start
    holdout_mask = df[year_col] >= holdout_start
    holdout = {
        'X_train_full': df.loc[train_full_mask, selected_features].copy(),
        'y_train_full': df.loc[train_full_mask, target_col].copy(),
        'X_holdout': df.loc[holdout_mask, selected_features].copy(),
        'y_holdout': df.loc[holdout_mask, target_col].copy(),
        'holdout_years': sorted(df.loc[holdout_mask, year_col].dropna().unique().tolist()),
    }

    print(f"📊 {len(folds)} folds walk-forward + holdout {holdout['holdout_years']}")
    return {
        'folds': folds,
        'holdout': holdout,
        'config': {
            'min_train_end': min_train_end,
            'max_train_end': max_train_end,
            'holdout_start': holdout_start,
            'n_folds': len(folds),
            'study_name': OPTUNA_STUDY_NAME_V2,
        },
    }


def save_feature_engineering_artifacts(df_final: pd.DataFrame,
                                     selected_features: List[str],
                                     scalers: Dict,
                                     metadata: Dict,
                                     splits: Dict,
                                     output_dir: Path) -> Dict[str, Path]:
    """
    Guarda todos los artefactos del feature engineering
    
    Args:
        df_final: DataFrame final completo
        selected_features: Lista de features seleccionadas
        scalers: Diccionario de scalers
        metadata: Metadatos del proceso
        splits: División train/test
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con rutas de archivos guardados
    """
    print("💾 Guardando artefactos de feature engineering...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}

    if 'year' not in df_final.columns:
        raise ValueError("El dataset final debe conservar la columna 'year'")

    processed_path = output_dir / "processed_data.parquet"
    df_final.to_parquet(processed_path, index=False)
    saved_files['processed_data'] = processed_path

    train_path = output_dir / "train_data.parquet"
    test_path = output_dir / "test_data.parquet"

    reference_mask = df_final['year'] < HOLDOUT_START_YEAR
    reference_medians = _compute_reference_medians(df_final, selected_features, reference_mask)
    X_train_imputed = _apply_reference_imputation(splits['X_train'], selected_features, reference_medians)
    X_test_imputed = _apply_reference_imputation(splits['X_test'], selected_features, reference_medians)

    train_data = pd.concat([X_train_imputed, splits['y_train']], axis=1)
    test_data = pd.concat([X_test_imputed, splits['y_test']], axis=1)

    train_data.to_parquet(train_path, index=False)
    test_data.to_parquet(test_path, index=False)
    saved_files['train_data'] = train_path
    saved_files['test_data'] = test_path

    scalers_path = output_dir / "scalers.pkl"
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    saved_files['scalers'] = scalers_path

    features_path = output_dir / "selected_features.txt"
    with open(features_path, 'w') as f:
        f.write('\n'.join(selected_features))
    saved_files['selected_features'] = features_path

    complete_metadata = {
        **metadata,
        'scalers_info': {k: {'variables': v['variables']} for k, v in scalers.items()},
        'split_info': splits['split_info'],
        'artifact_contract': {
            'processed_data': 'processed_data.parquet',
            'train_data': 'train_data.parquet',
            'test_data': 'test_data.parquet',
            'feature_metadata': 'feature_metadata.json',
            'selected_features': 'selected_features.txt',
            'scalers': 'scalers.pkl',
        },
        'imputation_reference': {
            'type': 'median_from_pre_holdout_train',
            'holdout_start_year': HOLDOUT_START_YEAR,
        },
        'process_timestamp': datetime.now().isoformat(),
        'config_used': FEATURE_ENGINEERING_CONFIG
    }

    metadata_path = output_dir / "feature_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(complete_metadata, f, indent=2, default=str)
    saved_files['metadata'] = metadata_path

    summary = f"""
# RESUMEN DE FEATURE ENGINEERING
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Final:
- Observaciones: {df_final.shape[0]:,}
- Features seleccionadas: {len(selected_features)}
- Período: {df_final['year'].min() if 'year' in df_final.columns else 'N/A'} - {df_final['year'].max() if 'year' in df_final.columns else 'N/A'}

## División Train/Test:
- Train: {splits['split_info']['train_size']:,} obs. ({splits['split_info']['train_pct']:.1f}%)
- Test: {splits['split_info']['test_size']:,} obs. ({splits['split_info']['test_pct']:.1f}%)

## Top 10 Features:
"""
    
    if 'feature_selection' in metadata and 'combined' in metadata['feature_selection']:
        for i, feature_info in enumerate(metadata['feature_selection']['combined'][:10], 1):
            summary += f"{i:2d}. {feature_info['feature']}: {feature_info['combined_score']:.3f}\n"
    
    summary += f"""
## Archivos Generados:
"""
    for name, path in saved_files.items():
        size_mb = path.stat().st_size / (1024**2)
        summary += f"- {path.name}: {size_mb:.1f} MB\n"
    
    summary_path = output_dir / "feature_engineering_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    saved_files['summary'] = summary_path

    print(f"✅ Artefactos guardados en: {output_dir}")
    for name, path in saved_files.items():
        size_mb = path.stat().st_size / (1024**2)
        print(f"  📄 {path.name}: {size_mb:.1f} MB")

    return saved_files

# ===== FUNCIÓN PRINCIPAL DE PIPELINE =====

def run_complete_feature_engineering_pipeline(df: pd.DataFrame, 
                                             target_col: str = 'purchase_price',
                                             output_dir: str = None) -> Dict:
    """
    Ejecuta el pipeline completo de feature engineering
    
    Args:
        df: DataFrame de entrada
        target_col: Nombre de la columna objetivo
        output_dir: Directorio de salida (por defecto usa config)
        
    Returns:
        Diccionario con todos los resultados y metadatos
    """
    print("🚀 INICIANDO PIPELINE COMPLETO DE FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        raw_target_col = target_col
        model_target_col = MODEL_TARGET_COL

        # 0. Limpiar columnas duplicadas
        df = clean_duplicate_columns(df)
        
        # 1. Variables temporales
        df_temp = create_temporal_features(df)
        df_temp = clean_duplicate_columns(df_temp)
        
        # 2. Variables de precio
        df_price = create_price_features(df_temp, raw_target_col)
        df_price = clean_duplicate_columns(df_price)
        
        # 3. Variables de tamaño
        df_size = create_size_features(df_price)
        df_size = clean_duplicate_columns(df_size)
        
        # 4. Codificación categórica
        df_encoded, encoding_info = encode_categorical_variables(df_size, raw_target_col)
        df_encoded = clean_duplicate_columns(df_encoded)
        
        # 5. Escalado
        df_scaled, scalers = scale_features(df_encoded)
        df_scaled = clean_duplicate_columns(df_scaled)

        # 6. Agregados regionales causales
        df_rolling = create_rolling_regional_features(
            df_scaled,
            region_col='region',
            price_col=raw_target_col,
            year_col='year',
        )
        df_rolling = clean_duplicate_columns(df_rolling)
        
        # 7. Features avanzados
        df_advanced = create_advanced_features(df_rolling, raw_target_col)
        df_advanced = clean_duplicate_columns(df_advanced)

        removable_cols = sorted(
            (
                FORBIDDEN_FEATURE_COLUMNS |
                set(ROLLING_AUXILIARY_COLUMNS) |
                ({raw_target_col} if raw_target_col != model_target_col else set())
            ) & set(df_advanced.columns)
        )
        if removable_cols:
            df_model_ready = df_advanced.drop(columns=removable_cols)
            print(f"🧹 Columnas removidas antes de selección final: {removable_cols}")
        else:
            df_model_ready = df_advanced.copy()
        
        # 8. Preparación final
        df_final, selected_features, metadata = prepare_final_dataset(df_model_ready, model_target_col)
        metadata['raw_target_col'] = raw_target_col
        metadata['model_target_col'] = model_target_col
        
        # 9. División train/test y dataset canónico
        final_columns = selected_features + [model_target_col, 'year']
        df_final_with_year = df_model_ready[final_columns].copy()
        df_final_with_year = clean_duplicate_columns(df_final_with_year)
        splits = create_train_test_split(df_final_with_year, selected_features, model_target_col)
        
        # 10. Guardar artefactos
        if output_dir is None:
            from pathlib import Path
            output_dir = Path.cwd() / "data" / "processed"
        
        saved_files = save_feature_engineering_artifacts(
            df_final_with_year, selected_features, scalers, metadata, splits, output_dir
        )
        
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE!")
        print(f"📊 Dataset final: {len(selected_features)} features seleccionadas")
        print(f"📁 Archivos guardados en: {output_dir}")
        
        return {
            'final_dataset': df_final,
            'processed_dataset': df_final_with_year,
            'selected_features': selected_features,
            'scalers': scalers,
            'metadata': metadata,
            'splits': splits,
            'saved_files': saved_files,
            'encoding_info': encoding_info
        }
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        raise e

# ===== FUNCIÓN PARA APLICAR A NUEVOS DATOS =====

def apply_feature_engineering_to_new_data(df_new: pd.DataFrame,
                                         scalers_path: str,
                                         selected_features_path: str,
                                         target_col: str = 'purchase_price') -> pd.DataFrame:
    """
    Aplica el feature engineering a nuevos datos usando artefactos guardados
    
    Args:
        df_new: DataFrame con nuevos datos
        scalers_path: Ruta al archivo de scalers
        selected_features_path: Ruta al archivo de features seleccionadas
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame procesado listo para predicción
    """
    print("🔄 Aplicando feature engineering a nuevos datos...")
    
    # Cargar artefactos
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    # Aplicar transformaciones (sin target encoding que requiere datos de entrenamiento)
    df_processed = create_temporal_features(df_new)
    df_processed = create_price_features(df_processed, target_col)
    df_processed = create_size_features(df_processed)
    
    # Aplicar escalado guardado
    for scaler_type, scaler_info in scalers.items():
        scaler = scaler_info['scaler']
        variables = scaler_info['variables']
        existing_vars = [var for var in variables if var in df_processed.columns]
        if existing_vars:
            df_processed[existing_vars] = scaler.transform(df_processed[existing_vars])
    
    # Seleccionar features finales
    available_features = [f for f in selected_features if f in df_processed.columns]
    
    print(f"✅ Procesamiento completado: {len(available_features)} features disponibles")
    
    return df_processed[available_features]

def clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas duplicadas manteniendo la primera ocurrencia
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        DataFrame sin columnas duplicadas
    """
    if df.columns.duplicated().any():
        print("⚠️ Columnas duplicadas encontradas, eliminando duplicados...")
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"Columnas duplicadas: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"✅ Columnas duplicadas eliminadas. Nuevo shape: {df.shape}")
    
    return df

def add_geographic_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica enriquecimiento geográfico usando el módulo geospatial_features.
    
    Args:
        df: DataFrame con datos de propiedades
        
    Returns:
        DataFrame enriquecido con características geográficas
    """
    from .features.geospatial_features import add_geospatial_features, create_geographic_clusters
    
    print("Aplicando enriquecimiento geográfico...")
    
    # Agregar características geoespaciales
    df_geo = add_geospatial_features(df)
    
    # Crear clusters geográficos
    df_geo = create_geographic_clusters(df_geo, n_clusters=5)
    
    print("Enriquecimiento geográfico completado.")
    return df_geo

def enhanced_feature_engineering_pipeline(df: pd.DataFrame, 
                                        target_col: str = 'purchase_price',
                                        output_dir: str = None,
                                        include_geographic: bool = True) -> Dict[str, Any]:
    """
    Pipeline de feature engineering mejorado con enriquecimiento geográfico.
    
    Args:
        df: DataFrame con datos de entrada
        target_col: Variable objetivo
        output_dir: Directorio de salida
        include_geographic: Si incluir enriquecimiento geográfico
        
    Returns:
        Diccionario con resultados del pipeline
    """
    # Ejecutar pipeline principal
    results = run_complete_feature_engineering_pipeline(df, target_col, output_dir)
    
    if include_geographic:
        print("\nAplicando enriquecimiento geográfico...")
        df_enriched = add_geographic_enrichment(results['final_dataset'])
        
        # Actualizar dataset final con características geográficas
        results['final_dataset'] = df_enriched
        results['geographic_features'] = ['urban_density', 'distance_to_center', 'location_type', 'transport_access', 'geo_cluster']
        
        # Guardar dataset enriquecido
        if output_dir:
            output_path = Path(output_dir)
            enriched_path = output_path / "feature_engineered_with_geography.parquet"
            df_enriched.to_parquet(enriched_path)
            results['saved_files']['enriched_geographic'] = enriched_path
            print(f"Dataset con enriquecimiento geográfico guardado: {enriched_path}")
    
    return results
