import sys
import pandas as pd
from pydantic import BaseModel, ValidationError, validator
from typing import Literal
import logging
from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class Target(BaseModel):
    condition: Literal[0, 1]


class Features(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

class Data(BaseModel):
    features: Features 
    target: Target 

    def get_dict(self):
        merged = {**self.dict()['features'], **self.dict()['target']}
        return merged

def gen_from_dataframe( data: pd.DataFrame):
    logger.info(f"dataframe for generation is {data}")
    features_data = data.drop(['condition'],axis=1)
    target_data = data[['condition']]
    logger.info(f"features for generation are {features_data}")
    logger.info(f"target for generation is {target_data}")
    logger.info(f"feature and target types are {type(features_data)}, {type(target_data)}")
    features = Features(**features_data.to_dict(orient='records')[0])
    target = Target(**target_data.to_dict(orient='records')[0])
    return Data(features=features, target=target)



@validator("age")
def reasonable_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age is not correct for alive human')
        return v

@validator("chol")
def reasonable_chol(cls, v):
        if v < 0 or v > 300:
            raise ValueError('cholesterol is not correct for alive human')
        return v


@validator("thalach")
def reasonable_thalach(cls, v):
        if v < 0 or v > 300:
            raise ValueError('heart rate is not correct for alive human')
        return v

@validator("oldpeack")
def reasonable_oldpeak(cls, v):
        if v < 0 or v > 8:
            raise ValueError('oldpeak is not correct for alive human')
        return v

