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
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "data", "feature_store")
FEATURE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

# =========================================================
# Fetch All Features
# =========================================================
def fetch_all_features():
    """Load latest feature store Parquet for model training"""
    # Find all versioned Parquets
    version_dirs = sorted(
        [os.path.join(FEATURE_STORE_DIR, "versions", d) for d in os.listdir(os.path.join(FEATURE_STORE_DIR, "versions"))],
        reverse=True
    )
    if not version_dirs:
        raise FileNotFoundError("❌ No versioned feature directories found")

    latest_dir = version_dirs[0]
    files = sorted([f for f in os.listdir(latest_dir) if f.endswith(".parquet")], reverse=True)
    if not files:
        raise FileNotFoundError("❌ No Parquet files found in latest version")

    latest_file = os.path.join(latest_dir, files[0])
    log_and_print(f"📥 Loading features from → {latest_file}")
    df = pd.read_parquet(latest_file)
    log_and_print(f"✅ Retrieved {df.shape[0]} rows × {df.shape[1]} features")
    return df

# =========================================================
# Store Features for Training
# =========================================================
def store_features(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(FEATURE_OUTPUT_DIR, f"heart_disease_features_{timestamp}.parquet")
    df.to_parquet(out_path, index=False)
    log_and_print(f"💾 Features stored for training → {out_path}")
    return out_path

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        df_all = fetch_all_features()
        log_and_print("🔍 Sample features:")
        log_and_print(df_all.head().to_string())

        store_features(df_all)
        log_and_print("🎯 Features ready for model training")
    except Exception as e:
        log_and_print(f"❌ Feature retrieval failed: {e}")
