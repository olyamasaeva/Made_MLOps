from typing import Optional

from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams

from marshmallow_dataclass import class_schema
import yaml

from custom_logs.log_decorator import log

@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    validation_dataset_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)

@log
def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))