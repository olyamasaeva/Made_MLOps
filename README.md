# Made-ML12-Masaeva-Olga
# Homework №2
### How to use model?
You can docker compose this model using command at the start folder:
```
docker compose up
```
or you can run command for building and running model:

```
docker build -t olyamasaeva/online_inference 
docker run -p 8000:8000 olyamasaeva/online_inference
```

How to download model:
```
docker pull olyamasaeva/mlops-server
```

### where is my server?

The service has adress _http://127.0.0.1:8000/docs_

### How to use online prediction model?
Open new terminal, from `online_inference/` run:
```
python3 online_inference/online_predict_pipeline.py path_to_config
```
where path_to_config: Yaml file - path to prediction config (examples are stored in config folder)

### Where are tests?
The docker is testing while building but if you want to run testing, use can do command:
```
python3 -m pytest online_inference/api/tests/test_predict.py
```
