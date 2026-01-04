import pandas as pd
import os
from datetime import datetime
import logging

# =========================================================
# Logging
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "feature_retrieval.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_and_print(msg):
    print(msg, flush=True)
    logging.info(msg)

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_STORE_PATH = os.path.join(
    BASE_DIR, "data", "feature_store", "heart_features.parquet"
)

FEATURE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

log_and_print("🔥 feature_retrieval.py started")

# =========================================================
# Fetch All Features (File-Based Feature Store)
# =========================================================
def fetch_all_features():
    """Loads all heart disease features for model training."""
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError("❌ Feature store file not found")

    log_and_print(f"📥 Loading features from feature store → {FEATURE_STORE_PATH}")
    df = pd.read_parquet(FEATURE_STORE_PATH)

    log_and_print(f"✅ Retrieved {df.shape[0]} rows × {df.shape[1]} features")
    return df

# =========================================================
# Store Features for Training
# =========================================================
def store_features(df):
    """Stores retrieved features with versioning for training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        FEATURE_OUTPUT_DIR,
        f"heart_disease_features_{timestamp}.parquet"
    )

    df.to_parquet(out_path, index=False)
    log_and_print(f"💾 Features stored → {out_path}")
    return out_path

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        df_all = fetch_all_features()
        log_and_print("🔍 Sample features:")
        log_and_print(df_all.head().to_string())

        stored_path = store_features(df_all)
        log_and_print(f"🎯 Features ready for model training → {stored_path}")

    except Exception as e:
        log_and_print(f"❌ Feature retrieval failed: {e}")
