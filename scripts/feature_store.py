import pandas as pd
import json
import os
import logging
from datetime import datetime

# =========================================================
# Logging
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "feature_store.log"),
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

TRANSFORMED_DIR = os.path.join(BASE_DIR, "data", "transformed")
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "data", "feature_store")
VERSION_DIR = os.path.join(FEATURE_STORE_DIR, "versions", "v1")

os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
os.makedirs(VERSION_DIR, exist_ok=True)

# =========================================================
# Load Transformed Data
# =========================================================
def load_transformed_data():
    path = os.path.join(TRANSFORMED_DIR, "heart_disease_transformed.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("❌ Transformed file not found")

    log_and_print(f"📥 Loading transformed data → {path}")
    return pd.read_parquet(path)

# =========================================================
# Store Features (FILE-BASED FEATURE STORE)
# =========================================================
def store_features(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    main_path = os.path.join(FEATURE_STORE_DIR, "heart_features.parquet")
    versioned_path = os.path.join(VERSION_DIR, f"heart_features_{timestamp}.parquet")

    df.to_parquet(main_path, index=False)
    df.to_parquet(versioned_path, index=False)

    log_and_print("✅ Features stored in feature store")
    log_and_print(f"📦 Latest → {main_path}")
    log_and_print(f"🕒 Versioned → {versioned_path}")

# =========================================================
# Store Feature Metadata
# =========================================================
def store_feature_metadata(df):
    metadata = {}

    descriptions = {
        "age": "Age of the patient",
        "sex": "Gender (1 = male, 0 = female)",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol level",
        "fbs": "Fasting blood sugar > 120 mg/dl",
        "restecg": "Resting ECG result",
        "thalch": "Maximum heart rate achieved",
        "exang": "Exercise induced angina",
        "oldpeak": "ST depression induced by exercise",
        "slope": "Slope of peak exercise ST segment",
        "ca": "Number of major vessels",
        "thal": "Thalassemia status",
        "age_risk": "Normalized age-based cardiac risk",
        "bp_risk": "High blood pressure risk",
        "chol_risk": "High cholesterol risk",
        "exercise_risk": "Exercise-induced angina risk",
        "st_severity": "ST depression severity",
        "vessel_risk": "Major vessel blockage risk",
        "cardiac_risk_score": "Composite cardiac risk score",
        "num": "Heart disease diagnosis (target)"
    }

    for col in df.columns:
        metadata[col] = {
            "description": descriptions.get(col, "Clinical feature"),
            "source": "Heart Disease Feature Engineering Pipeline",
            "version": 1,
            "created_at": datetime.now().isoformat()
        }

    meta_path = os.path.join(FEATURE_STORE_DIR, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    log_and_print(f"🧾 Feature metadata saved → {meta_path}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        df = load_transformed_data()
        store_features(df)
        store_feature_metadata(df)
        log_and_print("🎉 FEATURE STORE PIPELINE COMPLETED (NO DATABASE)")
    except Exception as e:
        log_and_print(f"❌ Pipeline failed: {e}")
