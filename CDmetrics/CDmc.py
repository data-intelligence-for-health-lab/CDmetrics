import numpy as np
import pandas as pd
from CDmetrics.nn import NN
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import multiprocessing


def worker(train_data_x, train_data_y, test_data_x, config):
    model = NN(config).train(train_data_x, train_data_y)
    prediction = model.predict(test_data_x)
    return prediction


def compute_metric(data, number_of_NNs, target_column, custom, resources):
    num_cpus = resources.get("CPU")

    max_hidden_neurons = round(len(data) * 0.01)
    Threshold = number_of_NNs * 0.9

    X_overall = data.drop(columns=[target_column])
    y_overall = data[target_column].values
    n_classes = len(set(y_overall))
    y_overall = (
        to_categorical(y_overall, num_classes=n_classes)
        if n_classes > 2
        else y_overall.reshape(-1, 1)
    )

    difficulity = []
    for index in tqdm(range(len(data))):

        X_test = X_overall.iloc[[index]]
        y_test = y_overall[index]
        y_test = np.argmax(y_test) if y_overall.shape[-1] > 1 else int(y_test)

        X = X_overall.drop(index=[index])
        y = np.delete(y_overall, index, axis=0)

        learn_rate = custom.get("learnRate", 0.01)
        batch_size = custom.get("batch_size", 64)
        activation = custom.get("activation", "relu")
        metric = custom.get("metric", "accuracy")
        validation_split = custom.get("validation_split", 0.3)
        epochs = custom.get("epochs", 100)

        params = {}
        params.update(
            {
                "learnRate": learn_rate,
                "batch_size": batch_size,
                "activation": activation,
                "num_classes": n_classes,
                "input_size": len(X.columns),
                "metric": metric,
                "validation_split": validation_split,
                "epochs": epochs,
            }
        )

        n_neureons_in_a_hidden_layer = 0
        correct_count = 0
        while n_neureons_in_a_hidden_layer < max_hidden_neurons:
            correct_count = 0
            n_neureons_in_a_hidden_layer += 1
            predictions = []
            with multiprocessing.Pool(processes=num_cpus) as pool:
                params["nn_architecture"] = [n_neureons_in_a_hidden_layer]
                async_results = [
                    pool.apply_async(worker, args=(X, y, X_test, params))
                    for _ in range(number_of_NNs)
                ]

                for result in async_results:
                    prediction_probabilities = result.get()
                    predictions.append(np.argmax(prediction_probabilities, axis=-1))

                predictions_flat = np.array(predictions).flatten()
                correct_count = np.sum(predictions_flat == y_test)

                if correct_count >= Threshold:
                    break

        difficulity.append(n_neureons_in_a_hidden_layer / max_hidden_neurons)

    return pd.DataFrame(difficulity)
