from pathlib import Path

# === Rutas base del proyecto ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
TABLES_DIR = RESULTS_DIR / "tablas"
CONFIG_FILE = PROJECT_ROOT / "src" / "config.py"

# === Archivos de datos ===
DATA_FILE = DATA_DIR / "raw" / "DKHousingPrices.parquet"
SAMPLE_FILE = DATA_DIR / "raw" / "DKHousingPricesSample100k.csv"

CLEAN_FILE = DATA_DIR / "processed" / "cleaned_data.parquet"
NULL_FILE = DATA_DIR / "anomalias" / "df_nulls.csv"

# === Configuración de H2O ===
H2O_URL = "http://localhost:54321"
DESTINATION_FRAME = "datos_h2o"

# === Columnas relevantes ===
TARGET = "purchase_price"
CATEGORICAL_COLS = ["region", "house_type", "sales_type"]
NUMERIC_COLS = ["sqm", "no_rooms", "year_build"]
DROP_COLS = ["house_id", "address", "sqm_price", "%_change_between_offer_and_purchase"]

# === Configuración de entrenamiento ===
N_TREES = 200
MAX_DEPTH = 10
LEARNING_RATE = 0.1

# === Flags de control ===
DEBUG = False
USE_GPU = True

ISDISTRIBUTED = True  # distribuir el h2o

DISTRIBUTED_DIR = Path("/mnt/sambashare/BigData-DATA/data")
DATA_DISTRIBUTED_PATH = DISTRIBUTED_DIR / "DKHousingPrices.parquet"
DISTRIBUTED_DATA_FILE = DISTRIBUTED_DIR / "DKHousingPrices.parquet"
DISTRIBUTED_SAMPLE_FILE = DISTRIBUTED_DIR / "DKHousingPricesSample100k.csv"
DISTRIBUTED_CLEAN_FILE = DISTRIBUTED_DIR / "processed" / "cleaned_data.parquet"
DISTRIBUTED_NULL_FILE = DISTRIBUTED_DIR / "anomalias" / "df_nulls.csv"

NULL_FILE = DATA_DIR / "anomalias" / "df_nulls.csv"

# ============================================
# CONFIGURACIÓN PARA MODELADO (REGRESIONES Y ÁRBOLES)
# Añadido manualmente desde feature_engineering.ipynb
# ============================================

# Rutas para datos procesados
PROCESSED_DATA_FILE = str(DATA_DIR / "processed" / "processed_data.parquet")
TRAIN_DATA_FILE = str(DATA_DIR / "processed" / "train_data.parquet")
TEST_DATA_FILE = str(DATA_DIR / "processed" / "test_data.parquet")
FEATURE_METADATA_FILE = str(DATA_DIR / "processed" / "feature_metadata.json")
SELECTED_FEATURES_FILE = str(DATA_DIR / "processed" / "selected_features.txt")
SCALERS_FILE = str(DATA_DIR / "processed" / "scalers.pkl")

TARGET = "log_price"  # Variable objetivo transformada

# Modelos para entrenamiento
REGRESSION_MODELS = [
    "LinearRegression", 
    "Ridge", 
    "Lasso", 
    "ElasticNet", 
    "SVR"
]

TREE_MODELS = [
    "DecisionTree", 
    "RandomForest", 
    "GradientBoosting", 
    "XGBoost", 
    "LightGBM"
]

# Información de datasets procesados
DATASET_INFO = {
    "processed_data": "Dataset completo con feature engineering aplicado",
    "train_data": "Conjunto de entrenamiento para modelado supervisado",
    "test_data": "Conjunto de prueba para evaluación final de modelos"
}

# Configuración específica para modelos de árboles
TREE_PARAMS = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "n_jobs": -1,
        "random_state": 42
    },
    "GradientBoosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": 42
    },
    "XGBoost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "tree_method": "auto",
        "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 128,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
}

# Configuración específica para modelos de regresión
REGRESSION_PARAMS = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0], 
        "l1_ratio": [0.1, 0.5, 0.9]
    },
    "SVR": {
        "kernel": ["linear", "poly", "rbf"],
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto"]
    }
}

# Métricas para evaluación
EVALUATION_METRICS = [
    "r2", 
    "rmse", 
    "mae", 
    "mape", 
    "medae"
]

# ============================================
# FEATURE ENGINEERING TEMPORAL (RFE-01)
# Ventana rolling causal para agregaciones regionales
# ============================================

ROLLING_WINDOW_YEARS = 3
"""Numero de anios previos [year - k, year - 1] para agregaciones regionales causales."""

MIN_OBS_PER_WINDOW = 20
"""Minimo de observaciones en la ventana para emitir valor; si no, NaN."""

# ============================================
# WALK-FORWARD CROSS-VALIDATION (RFE-01)
# Folds: train [1992, Y] / val [Y+1] para Y en [MIN, MAX]
# Holdout final: anios >= HOLDOUT_START_YEAR
# ============================================

CV_MIN_TRAIN_END_YEAR = 2013
CV_MAX_TRAIN_END_YEAR = 2021
HOLDOUT_START_YEAR = 2023

OPTUNA_STUDY_NAME_V2 = "h2o_xgb_walkforward_v1"
"""Nombre del study Optuna posterior al fix de leak."""

# ============================================
# MEDALLION ARCHITECTURE (MED-A1)
# Capas Bronze → Silver → Gold para pipeline analítico
# ============================================

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
HYPER_DIR = DATA_DIR / "hyper"
MEDALLION_METADATA_DIR = DATA_DIR / "medallion_metadata"

POSTAL_CENTROIDS_FILE = SILVER_DIR / "postal_centroids.parquet"
CPI_INDEX_FILE = SILVER_DIR / "cpi_index.parquet"
STL_COMPONENTS_FILE = SILVER_DIR / "stl_components.parquet"

# ============================================
# ANALYTICS CONVENTIONS (MED-A1)
# Constantes para KPIs del dashboard Tableau
# ============================================

CPI_BASE_YEAR = 2024
BASE_INDEX_YEAR = 1992
LOW_SAMPLE_THRESHOLD = 50
MACRO_SHOCK_YEARS = [1995, 2008, 2020, 2022]
FAMILY_SALE_TYPES = ["-", "Familiehandel"]

ROLLING_VOLATILITY_4Q = 4
ROLLING_VOLATILITY_8Q = 8
BOND_ELASTICITY_LAGS = [0, 1, 2]

DAWA_POSTNUMRE_URL = "https://api.dataforsyningen.dk/postnumre"

# ============================================
# PUENTE ML ↔ MEDALLION (UNI-01)
# El pipeline ML consume Bronze como fuente canónica limpia.
# ============================================

ML_INPUT_FILE = BRONZE_DIR / "transactions.parquet"
