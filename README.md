Post Truth Tools
===============================

version number: 0.0.1
author: Maksymilian Wojczuk

Overview
--------

Presented package is a set of tools for post-truth analysis. Nowadays many sites claim to provide 
reliable news while trying to be as much shocking and attracking as possible to draw attention
of customers. This set of tools has been created to help user distinguish between reliable sources
or articles and those that had been created only to seek intrest and stir up confusion and whats more 
even to influence political events or modify public opinion.

Installation
--------------------

To install use pip:

    $ pip install git+https://github.com/AGHPythonCourse2017/zad3-maxiwoj.git


Or clone the repo:

    $ git clone https://github.com/maxiwoj/post-truth-detector.git
    $ python setup.py install
    
Usage
------------

User is provided with a set of tools, all of them gathered in the "tools.py" file. The set contains:

### Clickbaitness
With this tool user may check title considering clickbaitness. Clickbaity titles are those, that 
do actually do not describe the real content of an article, but use psychological tricks to 
make user click into the link and to see the article. Sometimes such links come with additional 
unpredicted behaviours such as sharing its content on user's profile. 

To test clickbaitness use function which looks like: 
```python
def clickbaitness(heading):
    """Function tests using neural network model and most common phrases in 
    clickbaity titles for clickbaitness in headline of an article.
    provided model has 92 percent accuracy."""

```

### Site unreliability
With this tool user may check if the site on which he had found an article is a reliable source of 
information. This tool connects with an external server, that uses AI and neural networks to predict 
if the site is reliable.
```python
def site_unreliability(url):
    """Function for testing site unreliability"""
```
 
### Fact reliability
Sometimes user wants to check if the fact he has read is reliable. To do this use function:
```python
def fact_reliability(fact, number_of_searches=4):
    """simple function analising sources of google results for the fact 
    search
    
    Attributes: 
        fact -- string to check
        number_of_searches -- number of results from google taken under 
        consideration. Note, that every result is proceeded separately, 
        the more results taken, the more time the function will consume. max 
        number_of_searches is 10"""

```
This tool searches for the fact using google and checks reliability of sources that appear in the 
results. 

### Sentiment analisys
Not only intentional misleading in providing information is post_truthy. Sometimes the author 
is not objective and his article is more an opinion containing self-opinion-transformed facts than 
objective information. To test article's objectivity:
```python
def sentiment_analysis(text):
    """Simple function testing objectivity in the text"""
```

### Relativness analisys
Sometimes flashy and shocking headlines make user click the link but what is presented in the article 
does not correspond with the title. User may save time and check if the article is related to the 
title by using function:

```python
def relativeness_analisys(title, article):
    """function using neural networks to test for relativeness between title 
    and article
    
    Attributes:
        title -- string, title of an article
        article -- string, article to test relativness with title
        
    Return Value:
        0 if article is unrelated to the title 1 otherwise"""
```

Example
-------

#### clickbaitness
```python
import post_truth_detector as ptd
ptd.clickbaitness("The Way This Man Finds Out His Wife Has Been Cheating On Him Is Savage")
```

#### unreliability of sites
```python
import post_truth_detector as ptd
ptd.site_unreliability("http://edition.cnn.com/2017/06/02/football/champions-league-final-cardiff-security-real-madrid-juventus/index.html")
```

#### fact unreliability
```python
import post_truth_detector as ptd
ptd.fact_unreliability("Pope has a new baby")
```

#### Sentiment analisys
```python
import post_truth_detector as ptd
ptd.sentiment_analysis("I love this")
```

#### relativness
```python
import post_truth_detector as ptd
ptd.relativeness_analisys("Pope endorses Trump", "New model of Toyota has been announced last sunday")
```