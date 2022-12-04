import sys
import logging
import click
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from model.enities.predict_pipeline_params import read_predict_pipeline_params
from data.make_dataset import read_data, save_data
from model.model_gen.model_fit import open_model
from model.model_gen.model_predict import predict_model
import requests

from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def predict_pipeline(config_path: str):
    predict_params = read_predict_pipeline_params(config_path)
    logger.info(f"start {__name__} with params{predict_params}")
    features = read_data(predict_params.test_set_path).to_dict('records')
 #   features.drop(["trestbps"])
    logger.info(f"row feature value is equal to{json.dumps(features)}")
    predict = requests.post( 'http://127.0.0.1:8000/predict', json.dumps(features)).json()
    predicts = pd.DataFrame.from_records(predict)
    logger.info(f"predict is {predicts}")
    save_data(predicts, predict_params.predict_path)
    return 

@click.command(name="predict_online_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)

if __name__ == "__main__":
    predict_pipeline_command()
