import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from custom_logs.log_decorator import log

@log
def predict_model( model: Pipeline, features: pd.DataFrame) -> pd.Series:
    predicts = model.predict(features)
    return pd.Series(predicts)



