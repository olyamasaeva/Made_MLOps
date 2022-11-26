import logging
import sys
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from model.enities.split_params import SplittingParams

from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def read_data(path: str) -> pd.DataFrame:
    logger.info(f"reading path is {path}")
    data = pd.read_csv(path)
    logger.info(f"read data shape is {data.shape}")
    return data

@log
def save_data(data: pd.DataFrame, path:str):
    logger.info(f"saving data shape is {data.shape}")
    logger.info(f"saving data to {path}")
    data.to_csv(path, index=False, header=True) 

@log
def split_train_val_data( data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"dataframe shape is {data.shape}")
    logger.info(f"split coefficent is equal to {params.val_size}")
    train_data, val_data = train_test_split(data, test_size=params.val_size, random_state=params.random_state)
    logger.info(f"train dataframe shape is {train_data.shape}")
    logger.info(f"val dataframe shape is {val_data.shape}")
    return  train_data, val_data