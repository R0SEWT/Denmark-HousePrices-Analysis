from pathlib import Path

# === Rutas base del proyecto ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
TABLES_DIR = RESULTS_DIR / "tablas"

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