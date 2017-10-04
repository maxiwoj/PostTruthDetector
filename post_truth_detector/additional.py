import os

from googleapiclient.discovery import build


class RestApiException(Exception):
    def __init__(self, message):
        self.message = message


class BadArgumentException(Exception):
    def __init__(self, message):
        self.message = message


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    try:
        return res['items']
    except KeyError:
        print("Nothing found")
        return list()


relativness_model_path = os.path.dirname(__file__) \
                         + '/../models/relativeness_model_weights.joblib.pkl'
clickbait_model_weights_path = os.path.dirname(__file__) \
                               + "/../models/clickbait_model_weights"

clickbait_phrases_path = os.path.dirname(__file__) + "/../data/phrases.txt"
clickbait_words_path = os.path.dirname(__file__) + '/../data/words.txt'
clickbait_titles_path = os.path.dirname(__file__) + '/../data/clickbait.txt'
genuine_titles_path = os.path.dirname(__file__) + '/../data/genuine.txt'
