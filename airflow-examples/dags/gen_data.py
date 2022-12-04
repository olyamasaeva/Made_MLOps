from datetime import datetime
import os 
from airflow import DAG

from datetime import timedelta
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag =  DAG(
    'generate_data',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2022, 12, 3)
)
generate = DockerOperator(
    image='airflow-gen-data',
    command="--output='/data/raw/{{ ds }}'",
    network_mode='bridge',
    task_id='docker-airflow-gen-data',
    do_xcom_push=False,
    auto_remove=True,
    mounts=[Mount(source=os.environ.get("LOCAL_DATA_DIR"), target='/data', type='bind')]
)
generate