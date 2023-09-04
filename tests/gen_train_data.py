import sys
import pandas as pd
import numpy as np
from random import randint 


def gen_dataframe(dataframe_len):
    data = pd.DataFrame()
    for i in range(0, dataframe_len):
        data.loc[i,'age']= randint(0, 100)
        data.loc[i,'sex']= randint(0, 1)
        data.loc[i,'cp']= randint(0,3)
        data.loc[i,'trestbps']= randint(100, 200)	
        data.loc[i,'chol']= randint(180, 250)
        data.loc[i,'fbs']= randint(0, 1)
        data.loc[i,'restecg']= randint(0, 2)
        data.loc[i,'thalach']= randint(100, 200)
        data.loc[i,'exang']= randint(0, 1)
        data.loc[i,'oldpeak']= randint(0, 20)
        data.loc[i,'slope']= randint(0, 2)
        data.loc[i,'ca']= randint(0, 3)
        data.loc[i,'thal']= randint(0, 2)
        data.loc[i,'condition']= randint(0, 1)
    data = data.astype('int') 
    data.loc[np.random.choice(dataframe_len, dataframe_len // 3),'chol'] = np.nan
    data.loc[np.random.choice(dataframe_len, dataframe_len // 3),'oldpeak'] = np.nan
    data['chol'] = data['chol'].astype("Int32")
    data['oldpeak'] = data['oldpeak'].astype("Int32")
    return data

def main():
    data_size = int(sys.argv[1])
    file_name = "train_data_sample.csv"
    if len(sys.argv) >  2:
        file_name = sys.argv[2]
    data = gen_dataframe(data_size)
    data.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()