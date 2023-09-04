import numpy as np
import pandas as pd
import logging
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.enities.feature_params import FeatureParams
from pipeline.custom_logs.log_decorator import log


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@log
def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    cat_res =  pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())
    logger.info(f"cathegorical part is {cat_res.head()}")
    return cat_res

@log
def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline

@log
def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    num_res =  pd.DataFrame(num_pipeline.fit_transform(numerical_df))
    logger.info(f"numberical part is {num_res.head()}")
    return num_res

@log
def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),]
    )
    return num_pipeline


@log
def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    res =  pd.DataFrame(transformer.transform(df))
    logger.info(f"res head is {res.head()}")
    return res

@log
def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


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
