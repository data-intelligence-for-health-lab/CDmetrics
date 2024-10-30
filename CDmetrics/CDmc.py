import numpy as np
import pandas as pd
from CDmetrics.nn import NN


def compute_metric(data, number_of_NNs, target_column):
    max_hidden_units = round(len(data) * 0.01)
    Threshold = number_of_NNs * 0.9

    X_overall = data.drop(columns=[target_column], axis=1)
    y_overall = data[target_column]

    difficulity = []
    for index in range(len(data)):
        X_test = X_overall.iloc[[index]]
        y_test = y_overall[index]

        X = X_overall.drop(index=[index])
        y = y_overall.drop(index=[index])

        params = {}
        params.update(
            {
                "learnRate": 0.01,
                "batch_size": 32,
                "activation": "relu",
                "num_classes": len(set(y.values)),
                "input_size": len(X.columns),
            }
        )

        n_neureons_hidden_layer = 0
        count = 0
        while n_neureons_hidden_layer < max_hidden_units:
            if count < Threshold:
                n_neureons_hidden_layer += 1
                count = 0
                for _ in range(number_of_NNs):
                    params.update({"number_of_neurons": [n_neureons_hidden_layer]})
                    model = NN(params)
                    trained_model = model.train(X, y)
                    y_test = np.argmax(y_test)
                    y_pred = np.argmax(trained_model.predict(np.array(X_test)), axis=1)
                    if np.all(y_pred == y_test):
                        count += 1
                    else:
                        count += 0
            else:
                difficulity.append(n_neureons_hidden_layer / max_hidden_units)
                break

    return pd.DataFrame(difficulity)
