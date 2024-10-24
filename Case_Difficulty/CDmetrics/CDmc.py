import numpy as np
import pandas as pd
from nn import NN


def compute_metric(data, number_of_NNs, target_column, params):
    max_hidden_units = round(len(data) * 0.01)
    Threshold = number_of_NNs * 0.9

    X_overall = data.drop(columns=[target_column], axis=1)
    y_overall = data[target_column]

    difficulity = []
    for index in range(len(data)):
        # Test case to check the difficulty
        X_test = X_overall.iloc[[index]]
        y_test = y_overall[index]

        # The rest of cases to be used for training
        X = X_overall.drop(index=[index])
        y = y_overall.drop(index=[index])

        # NN having one hidden layer
        params["hidden_layer_sizes"] = 1
        model = NN(
            params,
        )
        # Increate number of neuron in the NNs
        # until it makes correct predictions than Threshold or reach the MNN
        n_neureons_hidden_layer = 0
        while n_neureons_hidden_layer < max_hidden_units:
            count = 0
            if count < Threshold:
                n_neureons_hidden_layer += 1
                count = 0
                # Generate number_of_NNs number of NN models
                for _ in range(number_of_NNs):
                    trained_model = model.train(n_neureons_hidden_layer, X, y)
                    y_test = np.argmax(y_test)
                    y_pred = np.argmax(trained_model.predict(np.array(X_test)), axis=1)

                    if np.all(y_pred == y_test):
                        count += 1
                    else:
                        count += 0
            else:
                difficulity.append(n_neureons_hidden_layer / max_hidden_units)

    return pd.DataFrame(difficulity)
