<img align="left" width="100" height="75" src="https://github.com/franckess/AzureML_Capstone/blob/main/img/microsoft-azure-640x401.png">
<img align="right" width="100" height="70" src="https://github.com/franckess/AzureML_Capstone/blob/main/img/mashable.jpeg">

# Online News Popularity Prediction


## Overview

A recent survey has shown that people read a fair bit amount of time online and the screen time is still rising. For that matter the number of views of specific content is of interest as more views translate to more revenues. Building a system to predict whether a news article will be popular or not can help editors to identify how they could improve their content but also how they could generate significant financial returns.

In this last project, we create two models to solve this classification problem: one using `Automated ML` and one customized model whose hyperparameters are tuned using `HyperDrive`. Then we compare the performance of both the models and deploy the best performing model as a __web service__.

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

![](https://github.com/franckess/AzureML_Capstone/blob/main/img/architecture_diagram.JPG)

## Dataset

### Overview

The dataset used in this project is a dataset made available on UCI Machine Learning Repository called [Online News Popularity Data Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#).

The dataset summarizes heterogeneous set of features about the articles published by Mashable between 2013 and 2015.

- Number of Records : 39,643
- Number of features : 61
- Target column : 1 

I will drop `url` and `timedelta` columns for further analysis since they do not have any predictive power.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
