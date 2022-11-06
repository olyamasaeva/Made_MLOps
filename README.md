# Made-ML12-Masaeva-Olga

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
For training set:
python ml_example/train_pipeline.py configs/train_config.yaml
For prediction:
python ml_example/predict_pipeline.py configs/predict_config.yaml
~~~


Project Organization
------------

    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metrics        <- Data from some models metrics
    │   ├── predicts       <- Data from some model predicts
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── make_dataset.py<- Code for datasets process
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for EDA and some prototyping
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── configs            <- Configs for pipelines
    │
    ├── predict_pipeline.py <- prediction pipeline file
    │
    ├── train_pipeline.py   <- training pipeline file
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── custom_logs        <- folder with logging information & code
    |
    ├── enities            <- enities code folder
    |
    ├── features           <- features code folder
    |
    ├── metrics            <- metrics class folder
    |
    ├── model_gen          <- metrics class folder
    |
    └──  models             <- serialize models storage
--------
