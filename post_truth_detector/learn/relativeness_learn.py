import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

bodies = pd.read_csv("../../data/train_bodies.csv")
stances = pd.read_csv("../../data/train_stances.csv")
relativness_model_filename = 'relativeness_model_weights.joblib.pkl'

MyTrainingListX = list()
MyTrainingListY = list()
for row in stances.iterrows():
    MyTrainingListX.append(
        [row[1][0], bodies[bodies["Body ID"] == row[1][1]].values[0][1]])
    MyTrainingListY.append(row[1][2])

logger = logging.getLogger(__name__)


def change(state):
    """Function to preprocess data from csv files"""
    if state == 'unrelated':
        return 0
    if state == 'discuss':
        return 1
    if state == 'disagree':
        return 1
    if state == 'agree':
        return 1


def count(article):
    """Function to count words from title occuring inside the article body"""
    a = 0
    heading = article[0].split()
    for word in heading:
        if word in article[1]:
            a += 1
    return a / len(heading)


def relativness_learn(log_level=Warning):
    """function training neural network to check relativness of the title 
    and article body"""
    logging.basicConfig(level=log_level)

    word_rate = np.array([count(x) for x in MyTrainingListX]).reshape(-1, 1)
    answer = np.array([change(x) for x in MyTrainingListY])

    train_x, test_x, train_y, test_y = train_test_split(word_rate, answer)

    scaler = StandardScaler()
    scaler.fit(train_x)

    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    model = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    logger.info(confusion_matrix(test_y, predictions))
    logger.info(classification_report(test_y, predictions))

    joblib.dump(model, relativness_model_filename, compress=9)
