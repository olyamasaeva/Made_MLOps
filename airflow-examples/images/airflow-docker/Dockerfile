ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}
ENV LOCAL_DATA_DIR = C:\Users\user\Desktop\made\MLOps\airflow-examples


RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==2.6.0 \
    apache-airflow-providers-google==7.0.0 \
    utils

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages
