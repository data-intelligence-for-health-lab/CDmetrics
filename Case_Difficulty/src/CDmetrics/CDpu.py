import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from CDmetrics.nn import NN
from CDmetrics.utils import tune_parameters


def compute_metric(
    data, target_column, number_of_predictions, max_layers, max_units, resources
):
    difficulity = []
    for index in tqdm(range(len(data))):
        test_data = data.iloc[[index]]
        test_data_x = test_data.drop(columns=[target_column], axis=1)
        test_data_y = test_data[target_column]
        train_data = data.drop(index=index, axis=0)

        best_config = tune_parameters(
            NN.tune, train_data, target_column, max_layers, max_units, resources
        )
        best_model = NN(best_config)
        predictions = []
        for i in range(number_of_predictions):
            prediction = best_model.model.predict(test_data_x, verbose=0).reshape(-1)
            predictions.append(prediction)

        if best_model.num_classes > 2:
            difficulity.append(
                multi_difficulty_formula(
                    predictions, test_data_y, NN.num_classes
                ).values
            )

        else:
            difficulity.append(
                binary_difficulty_formula(predictions, test_data_y).values
            )

    return pd.DataFrame(difficulity)


def binary_difficulty_formula(predictions, y_test):
    locations = abs(np.mean(predictions, axis=0) - y_test)
    distribution = np.std(predictions, axis=0) / math.sqrt(1 / 12)
    difficulty = locations + distribution / 2
    return difficulty


def multi_difficulty_formula(predictions, y_test, num_classes):
    one_hot_vector = [0] * num_classes
    one_hot_vector[y_test] = 1
    location = abs(np.mean(predictions, axis=0) - one_hot_vector)
    distribution = np.std(predictions, axis=0) / math.sqrt(1 / 12)
    class_difficulty = (location + distribution) / 2
    difficulty = np.mean(class_difficulty, axis=1)
    return difficulty
