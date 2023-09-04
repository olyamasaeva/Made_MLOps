import os
from typing import List

from py._path.local import LocalPath

from pipeline.train_pipeline import run_train_pipeline
from pipeline.predict_pipeline import run_predict_pipeline
from pipeline.enities.train_params import TrainingParams
from pipeline.enities.train_pipeline_params import TrainingPipelineParams
from pipeline.enities.split_params import SplittingParams
from pipeline.enities.feature_params import FeatureParams
from pipeline.enities.predict_pipeline_params import PredictPipelineParams

def test_train_e2e(
    model_path: str,
    tmpdir: LocalPath,
    dataset_path: str,  
    val_dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = model_path
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        validation_dataset_path = val_dataset_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
            use_log_trick=True,
        ),
        train_params=TrainingParams(model_type="RandomForest"),
    )

    real_model_path, metrics = run_train_pipeline(params)
    assert metrics.f1_score> 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)


def test_predict_e2e(
    tmpdir: LocalPath,
    val_dataset_path: str,
    model_path : str,
    prediction_path : str,
):
    params = PredictPipelineParams(
        model_path=model_path,
        test_set_path=val_dataset_path,
        predict_path=prediction_path
    )

    predicts = run_predict_pipeline(params)
    assert predicts.any() != None
    assert os.path.exists(params.predict_path)
