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
- `url`:                           URL of the article
- `timedelta`:                     Days between the article publication and the dataset acquisition
- `n_tokens_title`:                Number of words in the title
- `n_tokens_content`:              Number of words in the content
- `n_unique_tokens`:               Rate of unique words in the content
- `n_non_stop_words`:              Rate of non-stop words in the content
- `n_non_stop_unique_tokens`:      Rate of unique non-stop words in the content
- `num_hrefs`:                     Number of links
- `num_self_hrefs`:                Number of links to other articles published by Mashable
- `num_imgs`:                      Number of images
- `num_videos`:                    Number of videos
- `average_token_length`:          Average length of the words in the content
- `num_keywords`:                  Number of keywords in the metadata
- `data_channel_is_lifestyle`:     Is data channel 'Lifestyle'?
- `data_channel_is_entertainment`: Is data channel 'Entertainment'?
- `data_channel_is_bus`:           Is data channel 'Business'?
- `data_channel_is_socmed`:        Is data channel 'Social Media'?
- `data_channel_is_tech`:          Is data channel 'Tech'?
- `data_channel_is_world`:         Is data channel 'World'?
- `kw_min_min`:                    Worst keyword (min. shares)
- `kw_max_min`:                    Worst keyword (max. shares)
- `kw_avg_min`:                    Worst keyword (avg. shares)
- `kw_min_max`:                    Best keyword (min. shares)
- `kw_max_max`:                    Best keyword (max. shares)
- `kw_avg_max`:                    Best keyword (avg. shares)
- `kw_min_avg`:                    Avg. keyword (min. shares)
- `kw_max_avg`:                    Avg. keyword (max. shares)
- `kw_avg_avg`:                    Avg. keyword (avg. shares)
- `self_reference_min_shares`:     Min. shares of referenced articles in Mashable
- `self_reference_max_shares`:     Max. shares of referenced articles in Mashable
- `self_reference_avg_sharess`:    Avg. shares of referenced articles in Mashable
- `weekday_is_monday`:             Was the article published on a Monday?
- `weekday_is_tuesday`:            Was the article published on a Tuesday?
- `weekday_is_wednesday`:          Was the article published on a Wednesday?
- `weekday_is_thursday`:           Was the article published on a Thursday?
- `weekday_is_friday`:             Was the article published on a Friday?
- `weekday_is_saturday`:           Was the article published on a Saturday?
- `weekday_is_sunday`:             Was the article published on a Sunday?
- `is_weekend`:                    Was the article published on the weekend?
- `LDA_00`:                        Closeness to LDA topic 0
- `LDA_01`:                        Closeness to LDA topic 1
- `LDA_02`:                        Closeness to LDA topic 2
- `LDA_03`:                        Closeness to LDA topic 3
- `LDA_04`:                        Closeness to LDA topic 4
- `global_subjectivity`:           Text subjectivity
- `global_sentiment_polarity`:     Text sentiment polarity
- `global_rate_positive_words`:    Rate of positive words in the content
- `global_rate_negative_words`:    Rate of negative words in the content
- `rate_positive_words`:           Rate of positive words among non-neutral tokens
- `rate_negative_words`:           Rate of negative words among non-neutral tokens
- `avg_positive_polarity`:         Avg. polarity of positive words
- `min_positive_polarity`:         Min. polarity of positive words
- `max_positive_polarity`:         Max. polarity of positive words
- `avg_negative_polarity`:         Avg. polarity of negative  words
- `min_negative_polarity`:         Min. polarity of negative  words
- `max_negative_polarity`:         Max. polarity of negative  words
- `title_subjectivity`:            Title subjectivity
- `title_sentiment_polarity`:      Title polarity
- `abs_title_subjectivity`:        Absolute subjectivity level
- `abs_title_sentiment_polarity`:  Absolute polarity level
- `shares`:                        Number of shares (target)

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


![](https://i.pinimg.com/originals/e2/d7/c7/e2d7c71b09ae9041c310cb6b2e2918da.gif)