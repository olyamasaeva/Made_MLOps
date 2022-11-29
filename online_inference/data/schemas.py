from pydantic import BaseModel, ValidationError, validator
from typing import Literal


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

