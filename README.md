<img align="left" width="100" height="75" src="https://github.com/franckess/AzureML_Capstone/blob/main/img/microsoft-azure-640x401.png">
<img align="right" width="100" height="70" src="https://github.com/franckess/AzureML_Capstone/blob/main/img/mashable.jpeg">

# Online News Popularity Prediction


## Overview

A recent survey has shown that people read a fair bit amount of time online and the screen time is still rising. For that matter the number of views of specific content is of interest as more views translate to more revenues. Building a system to predict whether a news article will be popular or not can help editors to identify how they could improve their content but also how they could generate significant financial returns.

In this project we create two models to solve this classification problem: one using `Automated ML` and one customized model with hyperparameters tuned using `HyperDrive`. Then we compare the performance of both the models and deploy the best performing model. Finally the endpoint produced will be used to get some answers about predictions.

## Project Set Up and Installation

Before we get starting with this second project, it is important to set up our **local development environment** to match with the **Azure AutoML development environment**. 

Below are the steps:

1. Download and install `anaconda`
2. Open `anaconda CMD` in the new folder
3. Clone this repo
4. `cd` into the local directory
5. Run this `conda env create --file udacity_env.yml`
6. Run `jupyter notebook`

## Architecture Diagram

![](https://github.com/franckess/AzureML_Capstone/blob/main/img/architecture_diagram.jpeg)

## Dataset

### Overview

The dataset used in this project is a dataset made available on UCI Machine Learning Repository called [Online News Popularity Data Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#).

The dataset summarizes heterogeneous set of features about the articles published by Mashable between 2013 and 2015.

- Number of Instances: 39797
- Number of Attributes: 61 
    - 58 predictive attributes 
    - 2 non-predictive (`url` and `timedelta`) 
    - 1 target column

Attribute Information:
Column Names | Details
------------ | -------------
`url`|                           URL of the article
`timedelta`|                     Days between the article publication and the dataset acquisition
`n_tokens_title`|                Number of words in the title
`n_tokens_content`|              Number of words in the content
`n_unique_tokens`|               Rate of unique words in the content
`n_non_stop_words`|              Rate of non-stop words in the content
`n_non_stop_unique_tokens`|      Rate of unique non-stop words in the content
`num_hrefs`|                     Number of links
`num_self_hrefs`|                Number of links to other articles published by Mashable
`num_imgs`|                      Number of images
`num_videos`|                    Number of videos
`average_token_length`|          Average length of the words in the content
`num_keywords`|                  Number of keywords in the metadata
`data_channel_is_lifestyle`|     Is data channel 'Lifestyle'?
`data_channel_is_entertainment`| Is data channel 'Entertainment'?
`data_channel_is_bus`|           Is data channel 'Business'?
`data_channel_is_socmed`|        Is data channel 'Social Media'?
`data_channel_is_tech`|          Is data channel 'Tech'?
`data_channel_is_world`|         Is data channel 'World'?
`kw_min_min`|                    Worst keyword (min. shares)
`kw_max_min`|                    Worst keyword (max. shares)
`kw_avg_min`|                    Worst keyword (avg. shares)
`kw_min_max`|                    Best keyword (min. shares)
`kw_max_max`|                    Best keyword (max. shares)
`kw_avg_max`|                    Best keyword (avg. shares)
`kw_min_avg`|                    Avg. keyword (min. shares)
`kw_max_avg`|                    Avg. keyword (max. shares)
`kw_avg_avg`|                    Avg. keyword (avg. shares)
`self_reference_min_shares`|     Min. shares of referenced articles in Mashable
`self_reference_max_shares`|     Max. shares of referenced articles in Mashable
`self_reference_avg_sharess`|    Avg. shares of referenced articles in Mashable
`weekday_is_monday`|             Was the article published on a Monday?
`weekday_is_tuesday`|            Was the article published on a Tuesday?
`weekday_is_wednesday`|          Was the article published on a Wednesday?
`weekday_is_thursday`|           Was the article published on a Thursday?
`weekday_is_friday`|             Was the article published on a Friday?
`weekday_is_saturday`|           Was the article published on a Saturday?
`weekday_is_sunday`|             Was the article published on a Sunday?
`is_weekend`|                    Was the article published on the weekend?
`LDA_00`|                        Closeness to LDA topic 0
`LDA_01`|                        Closeness to LDA topic 1
`LDA_02`|                        Closeness to LDA topic 2
`LDA_03`|                        Closeness to LDA topic 3
`LDA_04`|                        Closeness to LDA topic 4
`global_subjectivity`|           Text subjectivity
`global_sentiment_polarity`|     Text sentiment polarity
`global_rate_positive_words`|    Rate of positive words in the content
`global_rate_negative_words`|    Rate of negative words in the content
`rate_positive_words`|           Rate of positive words among non-neutral tokens
`rate_negative_words`|           Rate of negative words among non-neutral tokens
`avg_positive_polarity`|         Avg. polarity of positive words
`min_positive_polarity`|         Min. polarity of positive words
`max_positive_polarity`|         Max. polarity of positive words
`avg_negative_polarity`|         Avg. polarity of negative  words
`min_negative_polarity`|         Min. polarity of negative  words
`max_negative_polarity`|         Max. polarity of negative  words
`title_subjectivity`|            Title subjectivity
`title_sentiment_polarity`|      Title polarity
`abs_title_subjectivity`|        Absolute subjectivity level
`abs_title_sentiment_polarity`|  Absolute polarity level
`shares`|                        Number of shares (target)


Class Distribution: the class value (shares) is continuously valued. We transformed the task into a binary task using a decision threshold of 1400.
Shares Value Range: `{'<1400':18490, '>=1400':21154}`

### Task

We want to know:

- How to predict which news articles will be popular
- What features about news articles make them more popular

This is important to:

- Help news sites become more profitable: Generate a model and feature insights that will give a company an advantage over other platforms vying for customer consumption.
- Raise awareness of important issues: Insights about what makes news popular can produce insights to help policy writers gain a following around their policy issue.

Trade-offs: Efficiency (popularity prediction) and fairness (even distribution of article post days and themes)

### Access
The original dataset was downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#). A subfolder `/data` was created to save this data.

The data is loaded to a remote datastore on AzureML from there we can apply both `AutoML` and `HyperDrive` approaches for modeling.

```python
# Create AML Dataset and register it into Workspace
example_data = 'https://github.com/franckess/AzureML_Capstone/blob/main/data/OnlineNewsPopularity.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)
# Create TabularDataset using TabularDatasetFactory
dataset = TabularDatasetFactory.from_delimited_files(path=example_data)
#Register Dataset in Workspace
dataset = dataset.register(workspace=ws, name=key, description=description_text)
```

## Automated ML

Azure Automated Machine Learning (`AutoML`) provides capabilities to automate iterative tasks of machine learning model development for given dataset to predict which article will be `popular` or not based on learnings from it's training data. In this approach, Azure Machine Learning taking user inputs such as `Dataset`, `Target Metric`, train multiple models (e.g. `Logistic Regression`, `Decision Tree`, `XGBoost`, etc.) and will return best performing model (or an ensemble) with highest training score achieved. We will train and tune a model using the `Accuray` primary metric for this project.

### AutoMLConfig

This class from Azure ML Python SDK represents configuration to submit an automated ML experiment in Azure ML. Configuration parameters used for this project includes:

Configration | Details | Value
------------ | ------------- | -------------
`compute_target` | Azure ML compute target to run the AutoML experiment on | compute_target
`task` | The type of task task to run, set as classification | classification
`training_data` | The training data to be used within the experiment contains training feature and a label column | Tabular Dataset
`label_column_name`	| The nae of the label column | 'label'
`path` | The full path to the Azure ML project folder	| './capstone-project'
`enable_early_stopping` | Enable AutoML to stop jobs that are not performing well after a minimum number of iterations	| True
`featurization` | Config indicator for whether featurization step should be done autometically or not	| auto
`debug_log ` | The log file to write debug information to | 'automl_errors.log'
`verbosity` | Reporting of run activity | logging.INFO

Also AutoML settings were as follows:

Configration | Details | Value
------------ | ------------- | -------------
`experiment_timeout_minutes` | Maximum amount of time in hours that all iterations combined can take before the experiment terminates | 60
`max_concurrent_iterations` | Represents the maximum number of iterations that would be executed in parallel | 9
`primary_metric` | The metric that the AutoML will optimize for model selection | accuracy

### Results

- Among all the models trained by AutoML, `Voting Ensemble` outperformed all the other models with `67.71% Accuracy`.

	- Ensemble models in Automated ML are combination of multiple iterations which may provide better predictions compared to a single iteration and appear as the final iterations of run.
	- Two types of ensemble methods for combining models: **Voting** and **Stacking**
	- Voting ensemble model predicts based on the weighted average of predicted class probabilities.

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_1.png)

Figure 1. Python SDK Notebook - AutoML Run Details widget

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_2.png)

Figure 2. Python SDK Notebook - Accuracy plot using AutoML Run Details widget

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_3.png)

Figure 3. Python SDK Notebook - Best performing run details

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_4.png)

Figure 4. Azure ML Studio - AutoML experiment completed

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_5.png)

Figure 5. Azure ML Studio - AutoML best perforing model summary

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_6.png)

Figure 6. Azure ML Studio - Performance Metrics of best performing model trained by AutoML


![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/automl_7.png)

Figure 7. Azure ML Studio - Models trained in multiple iterations using AutoML

### Hyperparameters generated for models ensembled in Voting Ensemble:
<details>
  <summary>Click to expand!</summary>

datatransformer
{'enable_dnn': None,
 'enable_feature_sweeping': None,
 'feature_sweeping_config': None,
 'feature_sweeping_timeout': None,
 'featurization_config': None,
 'force_text_dnn': None,
 'is_cross_validation': None,
 'is_onnx_compatible': None,
 'logger': None,
 'observer': None,
 'task': None,
 'working_dir': None}

prefittedsoftvotingclassifier
{'estimators': ['7', '0', '13', '14', '11', '30', '8', '5', '6', '19', '20'],
 'weights': [0.2,
             0.06666666666666667,
             0.13333333333333333,
             0.06666666666666667,
             0.06666666666666667,
             0.06666666666666667,
             0.13333333333333333,
             0.06666666666666667,
             0.06666666666666667,
             0.06666666666666667,
             0.06666666666666667]}

7 - maxabsscaler
{'copy': True}

7 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.4955555555555555,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_bin': 20,
 'max_depth': 7,
 'min_child_samples': 438,
 'min_child_weight': 6,
 'min_split_gain': 0.3157894736842105,
 'n_estimators': 600,
 'n_jobs': -1,
 'num_leaves': 224,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.2631578947368421,
 'reg_lambda': 0.42105263157894735,
 'silent': True,
 'subsample': 0.7426315789473684,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

0 - maxabsscaler
{'copy': True}

0 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'num_leaves': 31,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

13 - sparsenormalizer
{'copy': True, 'norm': 'max'}

13 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.7,
 'eta': 0.001,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 4,
 'max_leaves': 7,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0.3125,
 'reg_lambda': 1.875,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

14 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

14 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.6,
 'eta': 0.4,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 4,
 'max_leaves': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 2.0833333333333335,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.7,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

11 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

11 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.5,
 'eta': 0.1,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 15,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 2.0833333333333335,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

30 - sparsenormalizer
{'copy': True, 'norm': 'l1'}

30 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.7922222222222222,
 'importance_type': 'split',
 'learning_rate': 0.07368684210526316,
 'max_bin': 300,
 'max_depth': 4,
 'min_child_samples': 766,
 'min_child_weight': 6,
 'min_split_gain': 0.631578947368421,
 'n_estimators': 10,
 'n_jobs': -1,
 'num_leaves': 74,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0,
 'reg_lambda': 0.5263157894736842,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

8 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

8 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.7,
 'eta': 0.4,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 9,
 'max_leaves': 511,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 200,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.7708333333333335,
 'reg_lambda': 0.3125,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.6,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

5 - sparsenormalizer
{'copy': True, 'norm': 'l1'}

5 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.5,
 'eta': 0.1,
 'gamma': 0,
 'grow_policy': 'lossguide',
 'learning_rate': 0.1,
 'max_bin': 1023,
 'max_delta_step': 0,
 'max_depth': 10,
 'max_leaves': 0,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.875,
 'reg_lambda': 2.291666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.5,
 'tree_method': 'hist',
 'verbose': -10,
 'verbosity': 0}

6 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

6 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.9,
 'eta': 0.2,
 'gamma': 0.1,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 0,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 10,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1.3541666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.6,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

19 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

19 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 0.4955555555555555,
 'importance_type': 'split',
 'learning_rate': 0.0842121052631579,
 'max_bin': 260,
 'max_depth': 7,
 'min_child_samples': 2735,
 'min_child_weight': 1,
 'min_split_gain': 0.7368421052631579,
 'n_estimators': 25,
 'n_jobs': -1,
 'num_leaves': 140,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.9473684210526315,
 'reg_lambda': 0.2631578947368421,
 'silent': True,
 'subsample': 0.7921052631578948,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

20 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

20 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.7,
 'eta': 0.3,
 'gamma': 0,
 'grow_policy': 'lossguide',
 'learning_rate': 0.1,
 'max_bin': 1023,
 'max_delta_step': 0,
 'max_depth': 2,
 'max_leaves': 0,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 10,
 'n_jobs': -1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0.9375,
 'reg_lambda': 1.0416666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'hist',
 'verbose': -10,
 'verbosity': 0}
</details>

You can find the best AutoML model in the compressed file [automl_best_model.zip](https://github.com/franckess/AzureML_Capstone/blob/main/output/automl_model.pkl)

For more details about AutoML implementation check: [AutoML notebook](https://github.com/franckess/AzureML_Capstone/blob/main/02_aml-pipelines.ipynb)

## Hyperparameter Tuning
Classical models used for classification task are statistical models such as `Logistic Regression`. In this experiment I wanted to try a Machine Learning algorithm. I have chosen [`Light GBM (LGBM)`]( https://lightgbm.readthedocs.io/en/latest/index.html) for its great performance on different kind of tasks being, for instance, one of the most used algorithms in [Kaggle]( https://www.kaggle.com/) competitions.

The ranges of parameters for the LGBM used were chosen considering the parameters tuning guides for different scenarios provided [here]( https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).

`Bayesian sampling` method was chosen because tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric. This sampling method does not support `terminantion_policy`. Therefore, `policy=None`.

Steps required to tune hyperparameters using Azure ML's `HyperDrive package`;

1. Define the parameter search space using `Bayesian Parameter Sampling` 
2. Specify a `Accuracy` as a primary metric to optimize
3. Allocate `aml compute` resources
4. Launch an experiment with the defined configuration using `HyperDriveConfig`
5. Visualize the training runs with `RunDetails` Notebook widget
6. Select the best configuration for your model with `hyperdrive_run.get_best_run_by_primary_metric()`

In order to compare the performance of HyperDrive with the one of AutoML we chose as [objective metric]( https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) of LGBM `Accuracy`. For more information check this [link]( https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml#metric-normalization).

### Hyperparameter and search space strategy:

- `--num-leaves`: number of leaves of the tree
- `--min-data-in-leaf`: minimum # of samples in each leaf
- `--learning-rate`: learning rate
- `--feature-fraction`: ratio of features used in each iteration
- `--bagging-fraction`: ratio of samples used in each iteration
- `--bagging-freq`: bagging frequency
- `--max-depth` : to limit the tree depth explicitly

```python
param_sampling = BayesianParameterSampling(
    {
        "--num-leaves": quniform(8, 128, 1),
        "--min-data-in-leaf": quniform(20, 500, 10),
        "--learning-rate": choice(
            1e-4, 1e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1
        ),
        "--feature-fraction": uniform(0.1, 1),
        "--bagging-fraction": uniform(0.1, 1),
        "--bagging-freq": quniform(1, 30, 1),
        "--max-depth": quniform(5, 50, 5)
    }
)
```

### Results

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/hyperdrive_1.png)

Figure 9. Azure ML Studio Experiment submitted with HyperDrive from notebook

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/hyperdrive_2.png)

Figure 10. Python SDK Notebook: Monitor progress of run using Run Details widget

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/hyperdrive_3.png)

Figure 11. Python SDK Notebook: Best performing model from hyperparameter tuning using HyperDrive

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/hyperdrive_4.png)

Figure 12. Python SDK Notebook: HyperDrive Run Primary Metric Plot - Accuracy

Hyperparameters | Best Value
------------ | -------------
`--num-leaves` | 114
`--min-data-in-leaf` | 240 
`--learning-rate` | 0.05 
`--feature-fraction` 0.8767019272422398 
`--bagging-fraction` | 0.614723534867458 
`--bagging-freq` 27 
`--max-depth` 25

**Model Accuracy: 68.54%** which is better than the one generated using `AutoML` step.

You can find the best AutoML model in the compressed file [automl_best_model.zip](https://github.com/franckess/AzureML_Capstone/blob/main/output/lgb_model.pkl)

For more details about AutoML implementation check: [AutoML notebook](https://github.com/franckess/AzureML_Capstone/blob/main/01_Hyperparameter_tuning_final.ipynb)

## Deployment of the Best Model
`Deployment` is about delivering a trained model into production so that it can be consumed by others. By deploying a model you make it possible to interact with the HTTP API service and interact with the model by sending data over POST requests, for example.

Comparing the results of AutoML and HyperDrive we saw that HyperDrive gave us the best model (higher Accuracy). Therefore, this is the model to be deployed.

Details of the deployment of the model can be seen in section `Model Deployment` of the [HyperDrive notebook]( https://github.com/franckess/AzureML_Capstone/blob/main/01_Hyperparameter_tuning_final.ipynb).

Configuration object created for deploying an `AciWebservice` used for this project is as follows:

```python
script_file_name = './score.py'
inference_config = InferenceConfig(entry_script=script_file_name)

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 4, 
                                               tags = {'Company': "Mashable", 'Type': "Hyperdrive", "Version":"1"}, 
                                               description = 'sample service for Capstone Project Hyperdrive Classifier for Online News popularity')
aci_service_name = 'hyperdrive-deployment'
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(f'\nservice state: {aci_service.state}\n')
print(f'scoring URI: \n{aci_service.scoring_uri}\n')
print(f'swagger URI: \n{aci_service.swagger_uri}\n')
```

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/AciWebserice.png)

Figure 15. Python SDK Notebook: Deployment Completed 

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/successful_deployment.png)

Figure 16. Python SDK Notebook: Successful Deployment 

<!-- ## Screen Recording

Click [here](https://youtu.be/U2KGHlXrTfQ) to see a short demo of the project in action showing:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
 -->

## Standout Suggestions

### Consume model `endpoint.py`

Once the model is deployed, we use `scoring_uri` in [endpoint.py](https://github.com/franckess/AzureML_Capstone/blob/main/endpoint.py) script so we can interact with the trained model.

![Alt Text](https://github.com/franckess/AzureML_Capstone/blob/main/img/endpoint.png)

Figure 17. Endpoint consumption

<p align="center">
  <img src="https://media.giphy.com/media/beNeX29fBOizu/giphy.gif" alt="animated" />
</p>