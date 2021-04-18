# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import werkzeug
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({"n_tokens_title": pd.Series([0.0], dtype="float64"), "n_tokens_content": pd.Series([0.0], dtype="float64"), "n_unique_tokens": pd.Series([0.0], dtype="float64"), "num_hrefs": pd.Series([0.0], dtype="float64"), "num_self_hrefs": pd.Series([0.0], dtype="float64"), "num_imgs": pd.Series([0.0], dtype="float64"), "num_videos": pd.Series([0.0], dtype="float64"), "average_token_length": pd.Series([0.0], dtype="float64"), "num_keywords": pd.Series([0.0], dtype="float64"), "data_channel_is_entertainment": pd.Series([0], dtype="int64"), "data_channel_is_bus": pd.Series([0], dtype="int64"), "data_channel_is_socmed": pd.Series([0], dtype="int64"), "data_channel_is_tech": pd.Series([0], dtype="int64"), "data_channel_is_world": pd.Series([0], dtype="int64"), "kw_min_min": pd.Series([0.0], dtype="float64"), "kw_max_min": pd.Series([0.0], dtype="float64"), "kw_min_max": pd.Series([0.0], dtype="float64"), "kw_avg_max": pd.Series([0.0], dtype="float64"), "kw_min_avg": pd.Series([0.0], dtype="float64"), "kw_max_avg": pd.Series([0.0], dtype="float64"), "kw_avg_avg": pd.Series([0.0], dtype="float64"), "self_reference_min_shares": pd.Series([0.0], dtype="float64"), "self_reference_max_shares": pd.Series([0.0], dtype="float64"), "weekday_is_wednesday": pd.Series([0], dtype="int64"), "weekday_is_saturday": pd.Series([0], dtype="int64"), "weekday_is_sunday": pd.Series([0], dtype="int64"), "is_weekend": pd.Series([0], dtype="int64"), "LDA_00": pd.Series([0.0], dtype="float64"), "LDA_01": pd.Series([0.0], dtype="float64"), "LDA_02": pd.Series([0.0], dtype="float64"), "LDA_03": pd.Series([0.0], dtype="float64"), "LDA_04": pd.Series([0.0], dtype="float64"), "global_subjectivity": pd.Series([0.0], dtype="float64"), "global_sentiment_polarity": pd.Series([0.0], dtype="float64"), "global_rate_positive_words": pd.Series([0.0], dtype="float64"), "global_rate_negative_words": pd.Series([0.0], dtype="float64"), "rate_positive_words": pd.Series([0.0], dtype="float64"), "rate_negative_words": pd.Series([0.0], dtype="float64"), "avg_positive_polarity": pd.Series([0.0], dtype="float64"), "min_positive_polarity": pd.Series([0.0], dtype="float64"), "max_positive_polarity": pd.Series([0.0], dtype="float64"), "avg_negative_polarity": pd.Series([0.0], dtype="float64"), "min_negative_polarity": pd.Series([0.0], dtype="float64"), "max_negative_polarity": pd.Series([0.0], dtype="float64"), "title_subjectivity": pd.Series([0.0], dtype="float64"), "title_sentiment_polarity": pd.Series([0.0], dtype="float64"), "abs_title_sentiment_polarity": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'lgb_model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
