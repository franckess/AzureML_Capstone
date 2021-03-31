from sklearn.preprocessing import LabelEncoder
import argparse
import os
import pingouin as pg
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

def corr_drop_cols(df, corr_val = 0.75):
    df_copy = df.copy() # create a copy

    corrmat = pg.pairwise_corr(df_copy, method='pearson')[['X', 'Y', 'r']]
    df_corr = corrmat.sort_values(by='r', ascending=0)[(corrmat['r'] >= corr_val) | (corrmat['r'] <= -1*corr_va)]
    setcols = set(df_corr.Y.to_list())
    
    # Drop columns high correlation values
    df_copy = df_copy.drop(list(setcols), axis=1)

    return df_copy

def create_label(df):
    df_copy = df.copy() # create a copy

    df_copy['label'] = [1 if x >= 1400 else 0 for x in df_copy['shares']]

    # Encoding categorical data
    y = df_news['label'].values
    labelencoder = LabelEncoder()
    df_news['label'] = labelencoder.fit_transform(y)

if __name__ == "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", default=".", help="data folder mounting point")
    parser.add_argument("--num-leaves", type=int, dest="num_leaves", default=64, help="# of leaves of the tree")
    parser.add_argument("--min-data-in-leaf", type=int, dest="min_data_in_leaf", default=50, help="minimum # of samples in each leaf")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--feature-fraction",type=float,dest="feature_fraction",default=1.0,help="ratio of features used in each iteration")
    parser.add_argument("--bagging-fraction",type=float,dest="bagging_fraction",default=1.0,help="ratio of samples used in each iteration")
    parser.add_argument("--bagging-freq", type=int, dest="bagging_freq", default=1, help="bagging frequency")
    parser.add_argument("--max-rounds", type=int, dest="max_rounds", default=400, help="# of boosting iterations")
    args = parser.parse_args()
    args.feature_fraction = round(args.feature_fraction, 2)
    args.bagging_fraction = round(args.bagging_fraction, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

    # Data paths
    DATA_DIR = args.data_folder
  
    
    # Data and forecast problem parameters
    time_column_name = 'date'
    forecast_horizon = 28
    gap = 0

    
  # Parameters of GBM model
    params = {
        "objective": "root_mean_squared_error",
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "num_rounds": args.max_rounds,
        "early_stopping_rounds": 125,
        "num_threads": 16,
    }
    
    print(params)
    
    # Load training data
    default_train_file = os.path.join(DATA_DIR, "train.csv")
    if os.path.isfile(default_train_file):
        df_train = pd.read_csv(default_train_file,parse_dates=[time_column_name])
        print(df_train.head())
    else:
        df_train = pd.read_csv(os.path.join(DATA_DIR, "train_" + str(r + 1) + ".csv"),parse_dates=[time_column_name])
        
    # transform object type to category type to be used by lgbm
    df_train = change_data_type(df_train)
    
    # Split train data into training dataset and validation dataset
    df_train_2, df_val = split_train_test(df_train,forecast_horizon, gap)
    
    # Get features and labels
    X_train=df_train_2.drop(['demand'],axis=1)
    y_train=df_train_2['demand']
    X_val=df_val.drop(['demand'],axis=1)
    y_val=df_val['demand']
    
    X_train.drop(columns='date',inplace=True)
    X_val.drop(columns='date',inplace=True)
    
    d_train = lgb.Dataset(X_train, y_train)
    d_val = lgb.Dataset(X_val, y_val)
    
    print(X_train.info())
    
    # A dictionary to record training results
    evals_result = {}

    # Train LightGBM model
    bst = lgb.train(params, d_train, valid_sets=[d_train, d_val], categorical_feature="auto", evals_result=evals_result)

    # Get final training loss & validation loss 
    print(evals_result["training"].keys())
    train_loss = evals_result["training"]["rmse"][-1]
    val_loss = evals_result["valid_1"]["rmse"][-1]
    
    y_max = y_val.max()
    y_min = y_val.min()
    y_diff = (y_max - y_min)
    
    
    print("Final training loss is {}".format(train_loss/y_diff))
    print("Final test loss is {}".format(val_loss/y_diff))
    
    # Log the validation loss (NRMSE - normalized root mean squared error)
    run.log("NRMSE", np.float(val_loss/y_diff))
 
 
    #Dump the model using joblib
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=bst, filename='outputs/bst-model.pkl')