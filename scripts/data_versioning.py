import os
import subprocess
import logging

# =========================================================
# Logging Setup
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "heart_data_versioning.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_and_print(msg):
    print(msg, flush=True)
    logging.info(msg)

# =========================================================
# Project Paths (Heart Disease)
# =========================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRANSFORMED_DIR = "data/transformed"
FEATURE_STORE_DIR = "data/feature_store"
FEATURES_DIR = "data/features"

TRACK_DIRS = [
    RAW_DIR,
    PROCESSED_DIR,
    TRANSFORMED_DIR,
    FEATURE_STORE_DIR,
    FEATURES_DIR,
]

# =========================================================
# Shell Command Runner
# =========================================================
def run_command(command):
    """Runs shell commands safely with logging."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        logging.info(f"✅ SUCCESS: {command}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR: {command}\n{e.stderr}")
        return None

# =========================================================
# Remove from Git Tracking (if needed)
# =========================================================
def remove_git_tracking(path):
    """Stops Git from tracking large data folders."""
    if not os.path.exists(path):
        return

    tracked = run_command(f"git ls-files --error-unmatch {path}")
    if tracked is not None:
        run_command(f"git rm -r --cached {path}")
        run_command(f"git commit -m 'Stop tracking {path} in Git'")
        log_and_print(f"🧹 Removed {path} from Git tracking")

# =========================================================
# DVC Tracking
# =========================================================
def track_data_with_dvc():
    log_and_print("🚀 Starting Heart Disease data versioning")

    # Ensure DVC is initialized
    if not os.path.exists(".dvc"):
        run_command("dvc init")
        run_command("git commit -m 'Initialize DVC'")

    # Remove data folders from Git
    for folder in TRACK_DIRS:
        remove_git_tracking(folder)

    # Track folders with DVC
    for folder in TRACK_DIRS:
        if os.path.exists(folder):
            log_and_print(f"📦 Tracking {folder} with DVC")
            run_command(f"dvc add {folder}")
            run_command(f"git add {folder}.dvc")

    # Commit DVC metadata
    run_command("git commit -m 'Track heart disease datasets with DVC'")

    # Push to remotes
    run_command("git push origin main")
    run_command("dvc push")

    log_and_print("✅ Heart Disease data versioning completed successfully")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    track_data_with_dvc()
