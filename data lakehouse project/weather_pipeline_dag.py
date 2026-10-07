from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from datetime import datetime, timedelta

# Default settings applied to all tasks
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG (Directed Acyclic Graph)
with DAG(
    'weather_lakehouse_pipeline',
    default_args=default_args,
    description='Automated extraction and QA for the Weather Lakehouse',
    schedule_interval='0 2 * * *', # Cron expression: Runs every day at 2:00 AM
    start_date=datetime(2023, 10, 1),
    catchup=False,
    tags=['lakehouse', 'weather'],
) as dag:

    # Task 1: Execute the Python extraction script
    extract_to_bronze = BashOperator(
        task_id='extract_and_load_bronze',
        # In Docker, Airflow looks for scripts in the /opt/airflow/dags directory
        bash_command='python /opt/airflow/dags/extract_to_bronze.py' 
    )

    # Task 2: Data Quality Check on the Gold Layer
    # Fails the pipeline if the Gold view returns 0 rows, preventing silent failures
    check_gold_data = BigQueryCheckOperator(
        task_id='qa_check_gold_layer',
        sql='SELECT COUNT(*) FROM `open-meteo-analytics-platform.weather_lakehouse.gold_weather_features` HAVING COUNT(*) > 0',
        use_legacy_sql=False,
        gcp_conn_id='google_cloud_default' # Airflow's connection to your GCP account
    )

    # Define the execution order
    extract_to_bronze >> check_gold_data