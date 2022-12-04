import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from custom_logs.log_decorator import log
import logging
import sys

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def predict_model( model: Pipeline, features: pd.DataFrame) -> pd.Series:
    logger.info(f"model features are{features}")
    predicts = model.predict(features)
    return pd.Series(predicts)



