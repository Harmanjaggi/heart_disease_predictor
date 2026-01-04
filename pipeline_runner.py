import subprocess
import sys

SCRIPTS = [
    "scripts/ingest_data.py",
    "scripts/store_parquet.py",
    "scripts/data_validation.py",
    "scripts/data_preparation.py",
    "scripts/feature_store.py",
    "scripts/feature_retreival_storage.py",
    "scripts/data_versioning.py",
    "scripts/data_modeling.py"
]

def run(script):
    print(f"\n🚀 Running {script}")
    result = subprocess.run(
        ["python", script],
        check=True
    )

if __name__ == "__main__":
    for script in SCRIPTS:
        run(script)
    print("\n🎉 HEART DISEASE PIPELINE COMPLETED")
