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
DATA_FILE = DATA_DIR / "DKHousingPrices.parquet"
SAMPLE_FILE = DATA_DIR / "DKHousingPricesSample100k.csv"

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
PROCESSED_DATA_PATH = str(DATA_DIR / "processed" / "processed_data.parquet")
TRAIN_DATA_PATH = str(DATA_DIR / "processed" / "train_data.parquet")
TEST_DATA_PATH = str(DATA_DIR / "processed" / "test_data.parquet")
FEATURE_METADATA_PATH = str(DATA_DIR / "processed" / "feature_metadata.json")

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
