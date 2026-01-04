import os
import pandas as pd
import logging
import time

# =========================================================
# Paths
# =========================================================
CSV_FILE_PATH = "data/processed/heart_disease_uci.csv"
OUTPUT_FOLDER = "data/processed"
LOG_FOLDER = "logs"

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# =========================================================
# Ensure required directories exist
# =========================================================
def ensure_directories():
    """Create required directories if they do not exist."""
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

# Create directories before logging starts
ensure_directories()

# =========================================================
# Logging Configuration
# =========================================================
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "ingestion.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =========================================================
# Data Ingestion
# =========================================================
def ingest_data():
    """
    Reads a local CSV dataset, performs basic validation,
    and logs ingestion status with retry support.
    """
    attempts = 0

    while attempts < MAX_RETRIES:
        try:
            logging.info("🚀 Starting data ingestion process")

            # Check if file exists
            if not os.path.exists(CSV_FILE_PATH):
                raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")

            # Read CSV
            df = pd.read_csv(CSV_FILE_PATH)

            # Basic validation
            if df.empty:
                raise ValueError("Loaded dataset is empty")

            record_count = len(df)
            logging.info(f"✅ Successfully read {record_count} records")
            print(f"✅ Successfully read {record_count} records")

            return  # Exit on success

        except Exception as e:
            attempts += 1
            logging.exception(f"⚠️ Ingestion attempt {attempts} failed")
            print(f"⚠️ Ingestion attempt {attempts} failed: {e}")

            if attempts < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logging.critical("❌ Ingestion failed after maximum retry attempts")
    raise RuntimeError("Data ingestion failed after multiple retries")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    ingest_data()
