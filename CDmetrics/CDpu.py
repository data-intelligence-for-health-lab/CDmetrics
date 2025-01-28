import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from CDmetrics.nn import NN
from CDmetrics.utils import tune_parameters
from tensorflow.keras.utils import to_categorical
import multiprocessing


def worker(train_data_x, train_data_y, test_data_x, config):
    model = NN(config).train(train_data_x, train_data_y)
    prediction = model.predict(test_data_x)
    return prediction


def compute_metric(data, target_column, number_of_predictions, custom, resources):
    num_cpus = resources.get("CPU")
    difficulity = []
    data = data.reset_index(drop=True)
    for index in tqdm(range(len(data))):
        test_data = data.iloc[[index]]
        test_data_x = test_data.drop(columns=[target_column], axis=1)
        test_data_y = test_data[target_column].values

        train_data = data.drop(index=index, axis=0)
        train_data_x = train_data.drop(columns=[target_column], axis=1)
        train_data_y = train_data[target_column].values

        n_classes = len(set(data[target_column]))
        if n_classes > 2:
            test_data_y = to_categorical(test_data_y, num_classes=n_classes)
            train_data_y = to_categorical(train_data_y, num_classes=n_classes)

        best_config = tune_parameters(
            NN.tune, train_data, target_column, custom, resources
        )

        predictions = []
        with multiprocessing.Pool(processes=num_cpus) as pool:
            async_results = [
                pool.apply_async(
                    worker, args=(train_data_x, train_data_y, test_data_x, best_config)
                )
                for _ in range(number_of_predictions)
            ]
            for result in async_results:
                predictions.append(result.get())

        if n_classes > 2:
            difficulity.append(multi_difficulty_formula(predictions, test_data_y))

        else:
            difficulity.append(binary_difficulty_formula(predictions, test_data_y))

    return pd.DataFrame(difficulity)


def binary_difficulty_formula(predictions, y_test):
    locations = abs(np.mean(predictions, axis=0) - y_test)
    distribution = np.std(predictions, axis=0) / math.sqrt(1 / 12)
    difficulty = locations + distribution / 2
    return difficulty.item()


def multi_difficulty_formula(predictions, y_test):
    location = abs(np.mean(predictions, axis=0) - y_test)
    distribution = np.std(predictions, axis=0) / math.sqrt(1 / 12)
    class_difficulty = (location + distribution) / 2
    difficulty = np.mean(class_difficulty)
    return difficulty
