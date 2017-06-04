import unittest

from post_truth_detector import RestApiException, site_unreliability, \
    fact_unreliability, count, google_search, sentiment_analysis
from post_truth_detector.learn.relativeness_learn import map_state


class TestRemotes(unittest.TestCase):
    def test_site_unreliability(self):
        with self.assertRaises(RestApiException):
            site_unreliability("wwww.sdfdfg.pl")
        with self.assertRaises(RestApiException):
            site_unreliability("http:/www.wp.pl")
        assert round(site_unreliability("www.aszdziennik.pl")) == 1, \
            "Aszdziennik site unreliability"

    def test_fact_checker(self):
        self.assertEqual(round(fact_unreliability("Pope has a new baby")), 1,
                         "Bad fact check")
        self.assertEqual(round(fact_unreliability("Trump elected as the new "
                                                  "president")), 0,
                         "Real fact check")

    def test_google_connection(self):
        my_api_key = "AIzaSyBLyWxLJKU2Gydj7zRmA1mGsss_rERCDQA"
        my_cse_id = "005984777386616324044:zvaeoj2ttvu"

        self.assert_(google_search("Something to search for", my_api_key,
                                   my_cse_id, num=3), "test result not empty")


class TestCount(unittest.TestCase):
    def test_count(self):
        self.assertEqual(count(["ala ma kota", "ala jest bardzo fajna, "
                                               "ale nie ma kota"]), 1,
                         "Check number of words from title in article")

    def test_mapState(self):
        self.assertEqual(map_state("unrelated"), 0, "test unreleted map to 0")
        self.assertEqual(map_state("related"), 1, "test related map to 1")


class TestSentiments(unittest.TestCase):
    def test_sentimentals(self):
        self.assertGreater(sentiment_analysis("I love peanuts").subjectivity, 0)
        self.assertEqual(sentiment_analysis("Ala has a cat").subjectivity, 0)

if __name__ == '__main__':
    unittest.main()
