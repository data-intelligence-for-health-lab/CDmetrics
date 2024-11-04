import numpy as np
import pandas as pd
from CDmetrics.nn import NN
from tensorflow.keras.utils import to_categorical


def compute_metric(data, number_of_NNs, target_column):
    max_hidden_units = round(len(data) * 0.01)
    Threshold = number_of_NNs * 0.9

    X_overall = data.drop(columns=[target_column])
    y_overall = data[target_column].values  
    n_classes = len(set(y_overall))
    y_overall = to_categorical(y_overall, num_classes=n_classes) if n_classes > 2 else y_overall.reshape(-1, 1)

    difficulity = []
    for index in range(len(data)):
        X_test = X_overall.iloc[[index]]
        y_test = y_overall[index]
        y_test = np.argmax(y_test) if y_overall.shape[-1] > 1 else int(y_test)

        X = X_overall.drop(index=[index])
        y = np.delete(y_overall, index, axis=0)
        
        params = {}
        params.update(
            {
                "learnRate": 0.01,
                "batch_size": 32,
                "activation": "relu",
                "num_classes": n_classes,
                "input_size": len(X.columns),
            }
        )

        n_neureons_hidden_layer = 0
        count = 0
        while n_neureons_hidden_layer < max_hidden_units:
            count = 0
            n_neureons_hidden_layer += 1
            for _ in range(number_of_NNs):
                params["number_of_neurons"] = [n_neureons_hidden_layer]
                model = NN(params)
                trained_model = model.train(X, y)
                y_pred = np.argmax(trained_model.predict(np.array(X_test)), axis=-1)
                if np.all(y_pred == y_test):
                    count += 1
            if count >= Threshold:
                break
        difficulity.append(n_neureons_hidden_layer / max_hidden_units)
                
    return pd.DataFrame(difficulity)