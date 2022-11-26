FROM python:3.8.15-slim-buster

WORKDIR /online_interfence

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV SKLEARN_MODEL_PATH="model/models/model.pkl"


# install system dependencies
RUN apt-get update \
  #&& apt-get -y install netcat gcc postgresql \
  && apt-get clean

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install requests

COPY . /online_interfence/

RUN python3 -m pytest api/tests/test_predict.py
#CMD ["python3", "api/tests/test_predict.py"]