import requests
import json

# URL for the web service, should be similar to:
scoring_uri = 'http://5aad7139-c0fe-4249-a3e3-52fb7535b32e.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'uTsCm5fhEuVtoyqAM6xWWr2PnfIm8tVc'

# Two sets of data to score, so we get two results back
data = {"data": [{"n_tokens_title": 0.5238095238095238, "n_tokens_content": 0.13323105971206042, "n_unique_tokens": 0.0006325920646276748, "num_hrefs": 0.042763157894736836, "num_self_hrefs": 0.008620689655172414, "num_imgs": 0.0078125, "num_videos": 0.01098901098901099, "average_token_length": 0.6387343741908725, "num_keywords": 0.3333333333333333, "data_channel_is_entertainment": 0, "data_channel_is_bus": 0, "data_channel_is_socmed": 0, "data_channel_is_tech": 0, "data_channel_is_world": 1, "kw_min_min": 0.0, "kw_max_min": 0.002047587131367292, "kw_min_max": 0.012332503260998459, "kw_avg_max": 0.47818095576900277, "kw_min_avg": 0.5908604678089657, "kw_max_avg": 0.01182515012030831, "kw_avg_avg": 0.05734695866517057, "self_reference_min_shares": 0.001152614727854856, "self_reference_max_shares": 0.001152614727854856, "weekday_is_wednesday": 1, "weekday_is_saturday": 0, "weekday_is_sunday": 0, "is_weekend": 0, "LDA_00": 0.05393875832876898, "LDA_01": 0.3236042844264332, "LDA_02": 0.5982119195998494, "LDA_03": 0.05396522856706139, "LDA_04": 0.053930251835667194, "global_subjectivity": 0.49757416267899995, "global_sentiment_polarity": 0.3950170766327542, "global_rate_positive_words": 0.2449504159505239, "global_rate_negative_words": 0.16284486435072187, "rate_positive_words": 0.558441558442, "rate_negative_words": 0.441558441558, "avg_positive_polarity": 0.401116983791, "min_positive_polarity": 0.05, "max_positive_polarity": 0.8, "avg_negative_polarity": 0.6535364145660001, "min_negative_polarity": 0.0, "max_negative_polarity": 0.9285714285714, "title_subjectivity": 0.833333333333, "title_sentiment_polarity": 0.75, "abs_title_sentiment_polarity": 0.5}, {"n_tokens_title": 0.19047619047619047, "n_tokens_content": 0.06962473448194477, "n_unique_tokens": 0.000744605370379458, "num_hrefs": 0.046052631578947366, "num_self_hrefs": 0.034482758620689655, "num_imgs": 0.0859375, "num_videos": 0.0, "average_token_length": 0.5983764637748782, "num_keywords": 0.3333333333333333, "data_channel_is_entertainment": 0, "data_channel_is_bus": 0, "data_channel_is_socmed": 1, "data_channel_is_tech": 0, "data_channel_is_world": 0, "kw_min_min": 0.013227513227513227, "kw_max_min": 0.002550268096514745, "kw_min_max": 0.020277481323372468, "kw_avg_max": 0.2266097474208467, "kw_min_avg": 0.922283449140619, "kw_max_avg": 0.013643434767359248, "kw_avg_avg": 0.08472303532693719, "self_reference_min_shares": 0.0009024072097711372, "self_reference_max_shares": 0.020277481323372468, "weekday_is_wednesday": 0, "weekday_is_saturday": 0, "weekday_is_sunday": 0, "is_weekend": 0, "LDA_00": 0.8621355826953756, "LDA_01": 0.054075608761080377, "LDA_02": 0.05450107948911606, "LDA_03": 0.05421178467075573, "LDA_04": 0.05431893351240685, "global_subjectivity": 0.4905946255, "global_sentiment_polarity": 0.5780906501502268, "global_rate_positive_words": 0.4251246261220628, "global_rate_negative_words": 0.03666038920282912, "rate_positive_words": 0.9069767441859999, "rate_negative_words": 0.09302325581399999, "avg_positive_polarity": 0.374028749029, "min_positive_polarity": 0.0333333333333, "max_positive_polarity": 0.7, "avg_negative_polarity": 0.777083333333, "min_negative_polarity": 0.6, "max_negative_polarity": 0.875, "title_subjectivity": 1.0, "title_sentiment_polarity": 0.75, "abs_title_sentiment_polarity": 0.5}]}


# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())