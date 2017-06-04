from googleapiclient.discovery import build


class RestApiException(Exception):
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
