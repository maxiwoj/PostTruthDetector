from functools import reduce

from sklearn.externals import joblib
from textblob import TextBlob

from post_truth_detector.additional import RestApiException, google_search
from post_truth_detector.clickbaitness import Predictor
import requests
from post_truth_detector.learn.relativeness_learn import \
    relativness_model_filename, count


def clickbaitness(heading):
    """Function tests using neural network model and most common phrases in 
    clickbaity titles for clickbaitness in headline of an article."""
    predictor = Predictor()
    return predictor.predict(heading)


def site_unreliability(url):
    """Function for testing site unreliability"""
    if url.startswith("http://"):
        url = 'http://' + url.split('/')[2]
    elif url.startswith("https://"):
        url = 'https://' + url.split('/')[2]
    else:
        url = 'http://' + url.split('/')[0]

    params = {'url': url}
    response = requests.get("http://www.fakenewsai.com/detect",
                            params=params).json()
    if response.get('error'):
        raise RestApiException("Error while getting response from server or "
                               "badly formatted url")
    if response.get('result') is not None:
        return response.get('result')
    else:
        return response.get('fake')


def fact_reliability(fact, number_of_searches=4):
    """simple function analising sources of google results for the fact 
    search
    
    Attributes: 
        fact -- string to check
        number_of_searches -- number of results from google taken under 
        consideration. Note, that every result is proceeded separately, 
        the more results taken, the more time the function will consume. max 
        number_of_searches is 10"""
    my_api_key = "AIzaSyBLyWxLJKU2Gydj7zRmA1mGsss_rERCDQA"
    my_cse_id = "005984777386616324044:zvaeoj2ttvu"
    results = google_search(fact, my_api_key, my_cse_id, num=10)
    return reduce(lambda x, y: x + y, map(lambda result: site_unreliability(
        result['link']), results[:number_of_searches])) / number_of_searches


def sentiment_analysis(text):
    """Simple function testing objectivity in the text"""
    testimonials = TextBlob(text)
    return testimonials.sentiment


def relativeness_analisys(title, article):
    """function using neural networks to test for relativeness between title 
    and article
    
    Attributes:
        title -- string, title of an article
        article -- string, article to test relativness with title
        
    Return Value:
        0 if article is unrelated to the title 1 otherwise"""
    model = joblib.load(relativness_model_filename)
    return model.predict([count([title, article])])[0]
