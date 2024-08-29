import pandas as pd
import numpy as np
import statistics
import math
from nn import NN
from difficulty.metrics_utils.utils import tune_parameters
from sklearn.utils.multiclass import unique_labels


def binary_difficulty_formula(predictions, y_test):
    locations = abs(statistics.mean(predictions) - y_test)
    distribution = statistics.stdev(predictions) / math.sqrt(1 / 12)
    difficulty = (statistics.mean(locations) + distribution) / 2
    return difficulty


def multi_difficulty_formula(predictions, y_test):
    # predictions: [[0,0,1],[0,1,0],[0,1,0]]....,[0,0,1]] number_of_predictions
    num_classes = len(predictions[0])

    # one hot encoded y_test. e.g. 1-> [1,0,0] 2 -> [0,1,0] 3-> [0,0,1]
    one_hot_vector = [0] * num_classes
    one_hot_vector[y_test] = 1

    locations = []
    distributions = []
    class_difficulties = []

    for i in range(num_classes):
        # Get prediction probabilites for each class
        class_values = [sublist[i] for sublist in predictions]

        # Calculate the locaiton
        location = abs(statistics.mean(class_values) - one_hot_vector[i])
        locations.append(location)

        # Calculate the distribution
        distribution = statistics.stdev(class_values) / math.sqrt(1 / 12)
        distributions.append(distribution)

        # Calculate difficulty for each class
        class_difficulty = (location + distribution) / 2
        class_difficulties.append(class_difficulty)

    difficulty = statistics.mean(class_difficulties)
    return difficulty


def CDpu_main(data, target_column, number_of_predictions):

    X_overall = data.drop(columns=[target_column], axis=1)
    y_overall = data[target_column]

    difficulity = []
    for index in range(len(data)):
        # One test case to check the difficulty
        X_test = X_overall.iloc[[index]]
        y_test = y_overall[index]

        # The rest of cases to be used for train the model
        X = X_overall.drop(index=[index])
        y = y_overall.drop(index=[index])
        num_classes = len(unique_labels(y))

        # hyperparm Tuning
        best_config = NN.tune(tune_parameters(X, y))
        best_model = NN(best_config)
        predictions = []
        for i in range(number_of_predictions):
            prediction = best_model.predict(X_test)
            # Generate number_of_predictions number of predictions
            predictions.append(prediction)

        if num_classes > 2:
            difficulity.append(binary_difficulty_formula(predictions, y_test))
        else:
            difficulity.append(multi_difficulty_formula(predictions, y_test))

    return pd.DataFrame(difficulity)
