from pydantic import BaseModel, Field
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
    ca: Literal[0, 1, 2]
    thal: Literal[0, 1, 2]

class Data(BaseModel):
    features: Features
    target: Target

    
