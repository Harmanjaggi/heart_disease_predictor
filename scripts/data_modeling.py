import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

# =========================================================
# Paths
# =========================================================
FEATURES_DIR = "data/features/"
MODELS_DIR = "models/"
REPORTS_DIR = "reports/"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================================================
# MLflow Experiment
# =========================================================
mlflow.set_experiment("HeartDiseasePrediction")

# =========================================================
# Load Latest Feature File
# =========================================================
def get_latest_feature_file():
    files = sorted(os.listdir(FEATURES_DIR), reverse=True)
    for f in files:
        if f.endswith(".parquet"):
            return os.path.join(FEATURES_DIR, f)
    raise FileNotFoundError("❌ No feature files found")

def load_features():
    path = get_latest_feature_file()
    print(f"📥 Loading features from: {path}")
    return pd.read_parquet(path)

# =========================================================
# Train & Evaluate Models
# =========================================================
def train_model(df):

    # -----------------------------
    # Drop ID column
    # -----------------------------
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # -----------------------------
    # Target & Features
    # -----------------------------
    TARGET = "num"

    if TARGET not in df.columns:
        raise ValueError("❌ Target column 'num' not found")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X = X.astype("float64")

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    # -----------------------------
    # Training Loop
    # -----------------------------
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):

            print(f"🚀 Training {model_name}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics (macro for multi-class)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            # MLflow logging
            mlflow.log_param("model", model_name)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            # Signature
            signature = infer_signature(X_test, y_pred)

            # Save model locally
            model_path = os.path.join(
                MODELS_DIR, model_name.lower().replace(" ", "_") + ".pkl"
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name.lower().replace(" ", "_"),
                signature=signature,
                input_example=X_test.iloc[:1]
            )

            # Save report
            report_path = os.path.join(
                REPORTS_DIR, model_name.lower().replace(" ", "_") + ".txt"
            )
            with open(report_path, "w") as r:
                r.write(f"Model: {model_name}\n")
                r.write(f"Accuracy: {accuracy:.4f}\n")
                r.write(f"Precision (macro): {precision:.4f}\n")
                r.write(f"Recall (macro): {recall:.4f}\n")
                r.write(f"F1 Score (macro): {f1:.4f}\n")

            print(
                f"✅ {model_name} | Acc={accuracy:.4f} "
                f"Prec={precision:.4f} Rec={recall:.4f} F1={f1:.4f}"
            )

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    df_features = load_features()
    train_model(df_features)
