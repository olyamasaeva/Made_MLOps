FROM python:3.8.15-slim-buster
RUN python3 -m pip install --upgrade pip

COPY gen_data.py heart_cleveland_upload.csv requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "gen_data.py"]