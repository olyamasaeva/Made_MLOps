import json
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd

from data.make_dataset import read_data, split_train_val_data, save_data
from model.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from model.features.build_features import extract_target, drop_features

from model.model_gen.model_fit import(
    train_model,
    serialize_model,
    evaluate_model,
)
from custom_logs.log_decorator import log

from model.model_gen.model_predict import predict_model
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    result =  run_train_pipeline(training_pipeline_params)
    return result

@log
def run_train_pipeline(training_pipeline_params):
    logger.info(f"start train pipeline with params{training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(data, training_pipeline_params.splitting_params)

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(columns=training_pipeline_params.feature_params.target_col)
    val_df = val_df.drop(columns=training_pipeline_params.feature_params.target_col)

    train_df = drop_features(train_df, training_pipeline_params.feature_params)
    val_df = drop_features(val_df, training_pipeline_params.feature_params)

    save_data(val_df, training_pipeline_params.validation_dataset_path)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    model = train_model( train_df, train_target, training_pipeline_params.train_params )

    predicts = predict_model(model, val_df)

    metrics = evaluate_model(predicts, val_target)

    with open(training_pipeline_params.metric_path, "w+") as metric_file:
        json.dump(metrics.toJSON(), metric_file)
    logger.info(f"metrics are {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    return path_to_model, metrics

@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)

if __name__ == "__main__":
    train_pipeline_command()
