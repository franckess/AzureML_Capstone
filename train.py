from sklearn.preprocessing import LabelEncoder
import argparse
import os
import pingouin as pg
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
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
    y = df_copy['label'].values
    labelencoder = LabelEncoder()
    df_copy['label'] = labelencoder.fit_transform(y)

    return df_copy

def scaling_num(df):
    df_copy = df.copy() # create a copy

    col_list = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', \
          'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world', 'weekday_is_monday', \
          'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', \
          'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'timedelta', 'shares', 'label']

    # Scaling numerical data
    from sklearn.preprocessing import MinMaxScaler
    num_cols = [m for m in df_copy if m not in col_list]
    scale = MinMaxScaler()
    df_news[num_cols] = pd.DataFrame(scale.fit_transform(df_copy[num_cols].values), columns=[num_cols], index=df_copy.index)

    return df_copy

def feature_selection(df, OUT_DIR):
    df_copy = df.copy() # create a copy

    feat_selector = joblib.load(OUT_DIR)
    X = df_copy.drop(['shares', 'timedelta', 'label'], axis=1)
    keep_cols = list(X.columns[feat_selector.support_]) + ['timedelta','label']

    df_copy = df_copy[keep_cols]
    
    return df_copy

def split_train_test(df, qt = 0.8):
    df_copy = df.copy() # create a copy

    split_timedelta = df_copy['timedelta'].quantile(qt)

    train_df = df_copy[df_copy['timedelta'] <= split_timedelta]
    test_df = df_copy[df_copy['timedelta'] > split_timedelta]

    X_train = train_df.drop(['shares', 'timedelta', 'label'], axis=1)
    X_test = test_df.drop(['shares', 'timedelta', 'label'], axis=1)
    y_train = train_df['label']
    y_test = test_df['label']

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest='data_folder', default="https://raw.githubusercontent.com/franckess/AzureML_Capstone/main/data/OnlineNewsPopularity.csv", help="data folder")
    parser.add_argument("--boruta-model", type=str, dest='boruta_model', default="https://github.com/franckess/AzureML_Capstone/blob/main/output/boruta_model.pkl?raw=true", help="boruta folder")
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

    # Data path
    DATA_LOC = args.data_folder

    # Boruta path
    BORUTA_DIR = args.boruta_model

    
  # Parameters of GBM model
    params = {
        "objective": "binary",
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
    df = Dataset.Tabular.from_delimited_files(DATA_LOC)
        
    # Perform Data pre-processing
    df = corr_drop_cols(df)

    df = create_label(df)

    df = scaling_num(df)

    df = feature_selection(df, BORUTA_DIR)
    
    # Split train data into train & test
    X_train, X_test, y_train, y_test = split_train_test(df)
    
    d_train = lgb.Dataset(X_train, y_train)
    d_val = lgb.Dataset(X_test, y_test)

    # Train LightGBM model
    clf = lgb.train(params, d_train, 100, categorical_feature="auto")

    #prediction on the test set
    y_pred=clf.predict(X_test)
    
    
    # view accuracy
    accuracy=accuracy_score(y_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    
    # Log the validation loss (NRMSE - normalized root mean squared error)
    run.log("Model accuracy", np.float(accuracy_score(y_test, y_pred)))
 
 
    #Dump the model using joblib
    os.makedirs("./output/", exist_ok=True)
    model_path = os.path.join("./output/", 'lightgbm_model.pkl')
    joblib.dump(clf, filename=model_path)