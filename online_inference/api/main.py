import os
import sys
import logging
import pandas as pd
from fastapi import FastAPI, Depends, status
from data.schemas import Target, Features
from model.model_gen.model_fit import open_model
from model.model_gen.model_predict import predict_model

from fastapi import HTTPException, status
from custom_logs.log_decorator import log

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI()
app.sklearn_models = None

@app.on_event("startup")
def startup_event():
    app.sklearn_models = open_model(os.environ.get("SKLEARN_MODEL_PATH"))
    if app.sklearn_models == None:
         raise TypeError("None appeared instead of sklearn_models to open")

@app.post("/predict")
def predict(features: Features):
    dataframe_features =  pd.DataFrame([features.dict()])
    if app.sklearn_models == None:
        raise TypeError("None appeared instead of sklearn_models to predict")
    predicts = predict_model(app.sklearn_models, dataframe_features)
    return predicts.tolist()


@app.post('/health')
async def check_health(status_code=status.HTTP_200_OK):
    if app.sklearn_models == None:
          raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    print("model is ")
    print(app.sklearn_models)




