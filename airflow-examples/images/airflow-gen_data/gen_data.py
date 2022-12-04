import sys
import logging
import click
import json
import pandas as pd
import numpy as np
import requests


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def split_dataset(data: pd.DataFrame):
    features_data = data.drop(['condition'],axis=1)
    target_data = data[['condition']]
    return (features_data, target_data)


def gen_synt_data(output_folder : str):
    logger.info(f"output folder is {output_folder}")
    if output_folder == None:
        output_folder = '.'
    base_data = pd.read_csv("heart_cleveland_upload.csv")
    features_data, target_data = split_dataset(base_data)
    logger.info(f"features data is {features_data}")
    logger.info(f"target data is {target_data}")
 #   for i in range(len(base_data)):
 #       logger.info(f" features dict is {features_data.iloc[i].to_dict()}")
 #       logger.info(f" target dict is {target_data.iloc[i].to_dict()}")

    merged = [{'features' : (features_data.iloc[i].to_dict()), 'target' : (target_data.iloc[i].to_dict())} for i in range(len(base_data))] 
    logger.info(f"merged_data is {merged}")
    res = requests.post( 'http://0.0.0.0:8000/gen', json=merged,params={'sample_size' : 2} ).json()
    synt_data =  pd.DataFrame.from_records([{**a['features'], **a['target']} for a in res])
   # logger.info(f"response is{synt_data.json()}")
    logger.info(f"syn_data is {synt_data}")
    synt_feature, synt_target = split_dataset(synt_data)
    synt_feature.to_csv(output_folder + "data.csv",index=False)
    synt_target.to_csv(output_folder + "target.csv",index=False)


@click.command(name="gen_data")
@click.option('--output', type=click.Path(),
              help='Path to store train data')
def gen_synt_data_command(output:str):
   gen_synt_data(output)

if __name__ == "__main__":
    gen_synt_data_command()