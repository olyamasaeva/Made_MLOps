from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from custom_logs.log_decorator import log

@dataclass()
class PredictPipelineParams:
    model_path: "str"
    test_set_path: "str"
    predict_path: "str"


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)

@log
def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))