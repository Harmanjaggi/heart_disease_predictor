from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging
import os

# =========================================================
# Default arguments
# =========================================================
default_args = {
    'owner': 'harmansinghjaggi',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 22),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# =========================================================
# Base directory (PROJECT ROOT)
# =========================================================
# dags/heart_disease_dag.py
# └── ../ (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# =========================================================
# Helper function
# =========================================================
def run_script(script_path):
    """Runs a Python script and logs its output in Airflow."""
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(result.stdout)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)
        print(e.stderr)
        raise

# =========================================================
# DAG Definition
# =========================================================
dag = DAG(
    dag_id="heart_disease_dag",
    default_args=default_args,
    description="Heart disease prediction pipeline",
    schedule="@daily",
    catchup=False,
)

# =========================================================
# Tasks
# =========================================================
ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "ingest_data.py")],
    dag=dag,
)

store_parquet_task = PythonOperator(
    task_id='store_parquet',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "store_parquet.py")],
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "data_validation.py")],
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "data_preparation.py")],
    dag=dag,
)

feature_store_creation_task = PythonOperator(
    task_id='feature_store_creation',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "feature_store.py")],
    dag=dag,
)

feature_retreival_storage_task = PythonOperator(
    task_id='feature_retreival_storage',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "feature_retreival_storage.py")],
    dag=dag,
)

data_versioning_task = PythonOperator(
    task_id='data_versioning',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "data_versioning.py")],
    dag=dag,
)

data_modeling_task = PythonOperator(
    task_id='data_modeling',
    python_callable=run_script,
    op_args=[os.path.join(SCRIPTS_DIR, "data_modeling.py")],
    dag=dag,
)

# =========================================================
# Task Dependencies
# =========================================================
(
    ingest_task
    >> store_parquet_task
    >> validate_task
    >> prepare_data_task
    >> feature_store_creation_task
    >> feature_retreival_storage_task
    >> data_versioning_task
    >> data_modeling_task
)
