import numpy as np
import pandas as pd
import logging
import sys

from enities.feature_params import FeatureParams
from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    logger.info(f"targer_col names is {params.target_col}")
    target = df[params.target_col]
    logger.info("targer shape is {targer.shape}")
    return target

@log
def drop_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    logger.info(f"dropped features are {params.features_to_drop}")
    logger.info(f"dataframe columns names are {df.columns}")
    df = df.drop(columns=params.features_to_drop)
    return df
