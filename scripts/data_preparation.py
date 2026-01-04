import os
import pandas as pd
import numpy as np
import logging
import sys

import matplotlib
matplotlib.use("Agg")  # ✅ CI/CD safe (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# =========================================================
# Print + Log (flush safe)
# =========================================================
def log_and_print(msg):
    print(msg, flush=True)
    logging.info(msg)

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIS_DIR = os.path.join(BASE_DIR, "visualisation")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "data_preparation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log_and_print("🔥 data_preparation.py started")

# =========================================================
# Load CSV
# =========================================================
def get_latest_csv():
    csvs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("❌ No CSV files found")
    return max(csvs, key=os.path.getmtime)

def load_data():
    path = get_latest_csv()
    log_and_print(f"📥 Loading dataset: {path}")
    return pd.read_csv(path)

# =========================================================
# 📊 EDA VISUALIZATIONS
# =========================================================
def generate_eda_plots(df):
    log_and_print("📊 Generating EDA visualizations")

    # Target distribution
    plt.figure()
    df["num"].value_counts().plot(kind="bar")
    plt.title("Target Class Distribution (Heart Disease)")
    plt.xlabel("Disease Presence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "target_distribution.png"))
    plt.close()

    # Age distribution
    plt.figure()
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "age_distribution.png"))
    plt.close()

    # Chest pain vs target
    if "cp" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="cp", hue="num")
        plt.title("Chest Pain Type vs Heart Disease")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "cp_vs_target.png"))
        plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "correlation_heatmap.png"))
        plt.close()

    log_and_print(f"✅ EDA plots saved to {VIS_DIR}")

# =========================================================
# Prepare + Transform Data
# =========================================================
def prepare_data(df):
    log_and_print("🛠 Starting data preparation + transformation")

    # -----------------------------------------------------
    # EDA BEFORE TRANSFORMATION
    # -----------------------------------------------------
    generate_eda_plots(df)

    # -----------------------------------------------------
    # Drop ID
    # -----------------------------------------------------
    df.drop(columns=["id"], errors="ignore", inplace=True)

    # -----------------------------------------------------
    # Replace '?' with NaN
    # -----------------------------------------------------
    df.replace("?", np.nan, inplace=True)

    # -----------------------------------------------------
    # Normalize boolean-like columns
    # -----------------------------------------------------
    bool_map = {
        "TRUE": 1, "FALSE": 0,
        True: 1, False: 0,
        "true": 1, "false": 0
    }

    for col in ["fbs", "exang"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map)

    # -----------------------------------------------------
    # Encode sex
    # -----------------------------------------------------
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    # -----------------------------------------------------
    # Separate target
    # -----------------------------------------------------
    target = "num"
    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------------------------------
    # Drop fully-null columns
    # -----------------------------------------------------
    null_cols = X.columns[X.isna().all()].tolist()
    if null_cols:
        log_and_print(f"⚠ Dropping all-NaN columns: {null_cols}")
        X.drop(columns=null_cols, inplace=True)

    # -----------------------------------------------------
    # Identify column types
    # -----------------------------------------------------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # -----------------------------------------------------
    # Imputation
    # -----------------------------------------------------
    if numeric_cols:  # check not empty
        X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])
    if categorical_cols:  # check not empty
        X[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[categorical_cols])

    # -----------------------------------------------------
    # Encode categorical
    # -----------------------------------------------------
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    X["age_risk"] = X["age"] / X["age"].max()
    X["bp_risk"] = np.where(X["trestbps"] >= 140, 1, 0)
    X["chol_risk"] = np.where(X["chol"] >= 240, 1, 0)
    X["exercise_risk"] = X["exang"]
    X["st_severity"] = np.where(X["oldpeak"] >= 2.0, 1, 0)
    X["vessel_risk"] = np.where(X["ca"] > 0, 1, 0)

    risk_cols = [
        "age_risk", "bp_risk", "chol_risk",
        "exercise_risk", "st_severity", "vessel_risk"
    ]

    X["cardiac_risk_score"] = X[risk_cols].sum(axis=1)

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------
    exclude = risk_cols + ["cardiac_risk_score"]
    scale_cols = [c for c in X.columns if c not in exclude]

    if scale_cols:  # check not empty
        X[scale_cols] = MinMaxScaler().fit_transform(X[scale_cols])

    # -----------------------------------------------------
    # SMOTE
    # -----------------------------------------------------
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

    df_final = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns),
         pd.DataFrame(y_res, columns=[target])],
        axis=1
    )

    log_and_print("✅ Data preparation + transformation completed")
    return df_final

# =========================================================
# Save
# =========================================================
def save_data(df):
    out = os.path.join(DATA_DIR, "heart_disease_prepared.csv")
    df.to_csv(out, index=False)
    log_and_print(f"💾 Saved prepared data → {out}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        df = load_data()
        df_prepared = prepare_data(df)
        save_data(df_prepared)
        log_and_print("🎉 DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    except Exception as e:
        log_and_print(f"❌ Pipeline failed: {e}")
        sys.exit(1)
