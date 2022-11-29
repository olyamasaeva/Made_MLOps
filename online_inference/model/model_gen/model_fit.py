import pickle
import pandas as pd
import numpy as np
import logging
import sys
import os
import gdown

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score

from model.enities.train_params import TrainingParams
from model.metrics.metrics_class import MetricsClass

from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainingParams):
    logger.info(f"features shape is {features.shape}")
    logger.info(f"target shape is {target.shape}")
    logger.info(f"model type is {train_params.model_type}")
    if train_params.model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
            model = LogisticRegression(solver='liblinear')
    else:
            raise NotImplementedError()
    model.fit(features, target)
    logger.info("model is built!")
    return model

@log
def evaluate_model(predicts: pd.Series, target: pd.Series) -> MetricsClass:
    logger.info(f"target shape is {target.shape}")
    logger.info(f"predicts shape is {predicts.shape}")
    model_metrics = MetricsClass()
    model_metrics.accuracy = round(accuracy_score(predicts, target), 5)
    model_metrics.precision = round(precision_score(predicts, target), 5)
    model_metrics.recall = round(recall_score(predicts, target), 5)
    model_metrics.f1_score = round(f1_score(predicts, target), 5)
    model_metrics.roc_auc_score = round(roc_auc_score(predicts, target), 5)
    return model_metrics

@log
def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

@log
def open_model(model_path: str, from_net: bool=False) -> object:
    logger.info(f"model path is {model_path}")
    if from_net:
        gdown.download(model_path, os.environ.get("ONLINE_MODEL_DESTINATION"), quiet=False)
        model_path = os.environ.get("ONLINE_MODEL_DESTINATION")
    model = pickle.load(open(model_path,'rb'))
    logger.info(f"model is equal to {model}")
    if model == None:
        raise TypeError("Model not found")
    return model


