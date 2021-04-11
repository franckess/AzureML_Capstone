
import os
import pandas as pd
import json
import pickle
import logging 
import joblib

import pingouin as pg
import numpy as np
import requests
import pandas as pd
import azureml.core
import lightgbm as lgb
from io import BytesIO
from boruta import BorutaPy
from azureml.core.run import Run
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def init():
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'))
    print("Model path ", model_path)
    #load models
    deploy_model = joblib.load(model_path + '/lgb_model.pkl')
    
def corr_drop_cols(df, corr_val = 0.85):
    df_copy = df.copy() # create a copy
    corrmat = pg.pairwise_corr(df_copy, method='pearson')[['X', 'Y', 'r']]
    df_corr = corrmat.sort_values(by='r', ascending=0)[(corrmat['r'] >= corr_val) | (corrmat['r'] <= -1*corr_val)]
    setcols = set(df_corr.Y.to_list())
    # Drop columns high correlation values
    df_copy = df_copy.drop(list(setcols), axis=1)

    return df_copy

def create_label(df):
    df_copy = df.copy() # create a copy
    df_copy['label'] = [1 if x >= 1400 else 0 for x in df_copy['shares']]
    df_copy = df_copy.drop(['shares', 'timedelta'], axis=1)
    y = df_copy['label'].values
    labelencoder = LabelEncoder()
    df_copy['label'] = labelencoder.fit_transform(y)
    col_list = [s for s in df_copy.columns if 'is' in s]
    df_copy[col_list] = df_copy[col_list].apply(lambda x: labelencoder.fit_transform(x))

    return df_copy

def scaling_num(df):
    df_copy = df.copy() # create a copy
    from sklearn.preprocessing import MinMaxScaler
    col_list = [s for s in df_copy.columns if 'is' in s] + ['label']
    num_cols = [m for m in df_copy if m not in col_list]
    scale = MinMaxScaler()
    df_copy[num_cols] = pd.DataFrame(scale.fit_transform(df_copy[num_cols].values), columns=[num_cols], index=df_copy.index)

    return df_copy

def feature_selection(df, OUT_LOC):
    df_copy = df.copy() # create a copy
    mfile = BytesIO(requests.get(OUT_LOC).content) # BytesIO create a file object out of the response from GitHub 
    feat_selector = joblib.load(mfile)
    X = df_copy.drop(['label'], axis=1)
    keep_cols = list(X.columns[feat_selector.support_]) + ['label']
    df_copy = df_copy[keep_cols]

    return df_copy

def split_train_test(df):
    df_copy = df.copy() # create a copy
    X = df_copy.drop('label', axis=1)
    y = df_copy.pop('label')
    # Train-test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 100)

    return X_train, X_test, y_train, y_test

def run(data):
    # Boruta model location
    BORUTA_LOC = "https://github.com/franckess/AzureML_Capstone/releases/download/1.1/boruta_model_final.pkl"
    
    try:
        data.columns = data.columns.str.replace(' ','')
        data = data.drop(['url'], axis=1)
        data = corr_drop_cols(data)
        data = create_label(data)
        data = scaling_num(data)
        data = feature_selection(data, BORUTA_LOC)
        y = data.pop('label')
        X = data.drop(['label'], axis=1)
        
        result = deploy_model.predict(X)
        print("Result is ", result)
        return result.tolist()
    except Exception as e:
        error = str(e)
        prinrt("Error occured ", error)
        return error
