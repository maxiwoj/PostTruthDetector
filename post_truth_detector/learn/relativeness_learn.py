import logging
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from post_truth_detector.additional import relativness_model_path


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
    bodies = pd.read_csv(
        os.path.dirname(__file__) + "/../../data/train_bodies.csv")
    stances = pd.read_csv(os.path.dirname(__file__) +
                          "/../../data/train_stances.csv")

    my_training_list_x = list()
    my_training_list_y = list()
    for row in stances.iterrows():
        my_training_list_x.append(
            [row[1][0], bodies[bodies["Body ID"] == row[1][1]].values[0][1]])
        my_training_list_y.append(row[1][2])

    logger = logging.getLogger(__name__)

    logging.basicConfig(level=log_level)

    word_rate = np.array([count(x) for x in my_training_list_x]).reshape(-1, 1)
    answer = np.array([change(x) for x in my_training_list_y])

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

    joblib.dump(model, relativness_model_path)
