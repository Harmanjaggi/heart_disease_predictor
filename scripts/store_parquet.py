import os
import pandas as pd
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/storage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

RAW_CSV_PATH = "data/processed/heart_disease_uci.csv"  # Input CSV file
BASE_DIR = "data/processed/parquet/heart_disease/"         # Parquet storage base

DATE_PARTITION = datetime.now().strftime("%Y-%m-%d")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

TIMESTAMP_DIR = os.path.join(BASE_DIR, DATE_PARTITION)
OUTPUT_FILE = os.path.join(
    TIMESTAMP_DIR,
    f"heart_disease_raw_{TIMESTAMP}.parquet"
)

def create_directories():
    """Ensure necessary directories exist."""
    os.makedirs(TIMESTAMP_DIR, exist_ok=True)

def convert_to_parquet():
    """
    Reads the processed heart disease CSV,
    performs basic validation,
    and stores data as a partitioned Parquet file.
    """
    create_directories()

    # Check CSV existence
    if not os.path.exists(RAW_CSV_PATH):
        logging.error(f"CSV file not found: {RAW_CSV_PATH}")
        raise FileNotFoundError(f"CSV file not found: {RAW_CSV_PATH}")

    try:
        # Read CSV
        df = pd.read_csv(RAW_CSV_PATH)

        if df.empty:
            raise ValueError("Input dataset is empty")

        logging.info(f"Successfully read {len(df)} records from {RAW_CSV_PATH}")

        # Optional: limit rows (remove if full dataset needed)
        df = df.head(20000)

        # Write Parquet
        df.to_parquet(
            OUTPUT_FILE,
            index=False,
            engine="pyarrow"
        )

        logging.info(f"Parquet file stored at {OUTPUT_FILE}")
        print(f"✅ Parquet file created at: {OUTPUT_FILE}")

    except Exception as e:
        logging.exception("Error occurred during Parquet conversion")
        print(f"❌ Error converting to Parquet: {e}")
        raise

if __name__ == "__main__":
    convert_to_parquet()
