from __future__ import print_function
import os
import warnings
import argparse
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

warnings.filterwarnings(action='ignore', category=UserWarning, module='lightgbm')

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

    # Encoding target variable
    df_copy['label'] = [1 if x >= 1400 else 0 for x in df_copy['shares']]
    df_copy = df_copy.drop(['shares', 'timedelta'], axis=1)

    y = df_copy['label'].values
    labelencoder = LabelEncoder()
    df_copy['label'] = labelencoder.fit_transform(y)

    # Encoding all categorical variables
    col_list = [s for s in df_copy.columns if 'is' in s]
    df_copy[col_list] = df_copy[col_list].apply(lambda x: labelencoder.fit_transform(x))

    return df_copy

def scaling_num(df):
    df_copy = df.copy() # create a copy

    # Scaling numerical data
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

if __name__ == "__main__":
    
    print('azureml.core.VERSION={}'.format(azureml.core.VERSION))
    # Parse ainput arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-leaves", type=int, dest="num_leaves", default=64, help="# of leaves of the tree")
    parser.add_argument("--min-data-in-leaf", type=int, dest="min_data_in_leaf", default=50, help="minimum # of samples in each leaf")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--feature-fraction",type=float,dest="feature_fraction",default=1.0,help="ratio of features used in each iteration")
    parser.add_argument("--bagging-fraction",type=float,dest="bagging_fraction",default=1.0,help="ratio of samples used in each iteration")
    parser.add_argument("--bagging-freq", type=int, dest="bagging_freq", default=1, help="bagging frequency")
    parser.add_argument("--max-depth", type=int, dest="max_depth", default=20, help="To limit the tree depth explicitly")
    args = parser.parse_args()
    args.feature_fraction = round(args.feature_fraction, 2)
    args.bagging_fraction = round(args.bagging_fraction, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

    # Data location
    DATA_LOC = "https://github.com/franckess/AzureML_Capstone/blob/main/data/OnlineNewsPopularity.csv?raw=true"

    # Boruta model location
    BORUTA_LOC = "https://github.com/franckess/AzureML_Capstone/releases/download/1.1/boruta_model_final.pkl"

    
  # Parameters of GBM model
    params = {
        "objective": "binary",
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "num_threads": 16,
        "max_depth": args.max_depth
    }
    
    print(params)
    
    # Load training data
    print('Loading data from {}'.format(DATA_LOC))
    df = pd.read_csv(DATA_LOC)

    # Removing space character in the feature names
    df.columns = df.columns.str.replace(' ','')

    # Drop URL column
    df = df.drop(['url'], axis=1)
        
    # Perform Data pre-processing
    print('Running pre-processing steps')
    df = corr_drop_cols(df)

    df = create_label(df)

    df = scaling_num(df)

    print('Loading Boruta model for feature selection from {}'.format(BORUTA_LOC))
    df = feature_selection(df, BORUTA_LOC)
    
    # Split train data into train & test
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Train LightGBM model
    print('Running modeling step')
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)

    #prediction on the test set
    y_pred=clf.predict(X_test)
    
    # view accuracy
    accuracy=accuracy_score(y_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))
    
    # Log the validation loss (NRMSE - normalized root mean squared error)
    run.log("Accuracy", np.float(accuracy))
 
 
    #Dump the model using joblib
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(value=clf, filename="outputs/lgb_model.pkl")