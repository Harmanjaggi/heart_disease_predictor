import os
import pandas as pd
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/data_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = "data/processed/"
REPORT_PATH = "reports/data_quality_report.csv"

def get_latest_csv():
    """Fetch latest processed CSV file."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Data directory not found: {DATA_DIR}")

    csv_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("❌ No processed CSV files found")

    return max(csv_files, key=os.path.getmtime)


def load_data():
    """Load processed heart disease dataset."""
    csv_path = get_latest_csv()
    logging.info(f"Loading dataset: {csv_path}")
    print(f"✅ Loading dataset: {csv_path}")
    return pd.read_csv(csv_path)

def check_missing_values(df):
    """Check missing values."""
    missing = df.isnull().sum()
    return missing[missing > 0]


def check_data_types(df):
    """Validate expected schema."""
    expected_types = {
        "id": "int64",
        "age": "int64",
        "sex": "object",
        "dataset": "object",
        "cp": "object",
        "trestbps": "int64",
        "chol": "int64",
        "fbs": "object",        # TRUE / FALSE
        "restecg": "object",
        "thalch": "int64",
        "exang": "object",      # TRUE / FALSE
        "oldpeak": "float64",
        "slope": "object",
        "ca": "int64",
        "thal": "object",
        "num": "int64",         # target
    }

    issues = {}
    for col, expected in expected_types.items():
        if col in df.columns and str(df[col].dtype) != expected:
            issues[col] = {
                "actual": str(df[col].dtype),
                "expected": expected,
            }
    return issues


def check_duplicates(df):
    """Check duplicate records."""
    return df.duplicated().sum()


def check_target_distribution(df):
    """Check class balance of target."""
    if "num" not in df.columns:
        return {}

    return df["num"].value_counts().sort_index().to_dict()

def generate_quality_report(df):
    """Generate data quality report."""
    os.makedirs("reports", exist_ok=True)

    missing = check_missing_values(df)
    dtype_issues = check_data_types(df)
    duplicates = check_duplicates(df)
    target_dist = check_target_distribution(df)

    report = {
        "Metric": [
            "Missing Values",
            "Data Type Issues",
            "Duplicate Rows",
            "Target Classes",
        ],
        "Details": [
            int(missing.sum()) if not missing.empty else 0,
            len(dtype_issues),
            duplicates,
            str(target_dist),
        ],
    }

    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_PATH, index=False)

    logging.info("Data quality report generated successfully")
    print(f"✅ Data Quality Report saved at: {REPORT_PATH}")

if __name__ == "__main__":
    df = load_data()
    generate_quality_report(df)
