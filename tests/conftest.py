import os

import pytest
from typing import List

@pytest.fixture()
def prediction_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "fake_data/sample_prediction.pkl")


@pytest.fixture()
def model_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "sample_model.pkl")

@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "fake_data/train_data_sample.csv")


@pytest.fixture()
def val_dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "fake_data/val_data_sample.csv")

@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
    "age",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["trestbps"]