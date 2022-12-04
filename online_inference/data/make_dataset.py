import logging
import sys
import pandas as pd
from typing import Tuple
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

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
    data.to_csv(path, index=False, header=False) 

@log
def gen_data(example_data: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
     logger.info(f"example_data is {example_data}")
     #should better to hardcode it somewhere - ToDo
     dataset_path = 'data/raw/sample_data.csv'
     mode = 'independent_attribute_mode'
     threshold_value = 20
     categorical_attributes = {'sex': True, 'cp': True, 'fbs': True,  'restecg': True, 'exang' : True, 'slope' : True, 'ca' :  True, 'thal' : True, 'condition': True}
     candidate_keys = {'age': False, 'chol' : False, 'thalach' : False}     
     description_file = 'data/raw/description_file'
     example_data.to_csv(dataset_path)
     describer = DataDescriber(category_threshold=threshold_value)
     describer.describe_dataset_in_independent_attribute_mode(dataset_file=dataset_path,
                                                         attribute_to_is_categorical=categorical_attributes,
                                                         attribute_to_is_candidate_key=candidate_keys)
     describer.save_dataset_description_to_file(description_file)
     generator = DataGenerator()
     generator.generate_dataset_in_independent_mode(sample_size, description_file)
     ans =  pd.DataFrame(generator.synthetic_dataset)
     logger.info(f"generated dataset is {ans}")
     return ans

@log
def split_train_val_data( data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"dataframe shape is {data.shape}")
    logger.info(f"split coefficent is equal to {params.val_size}")
    train_data, val_data = train_test_split(data, test_size=params.val_size, random_state=params.random_state)
    logger.info(f"train dataframe shape is {train_data.shape}")
    logger.info(f"val dataframe shape is {val_data.shape}")
    return  train_data, val_data