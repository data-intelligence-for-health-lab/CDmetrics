import pandas as pd
from tqdm import tqdm
from CDmetrics.utils import tune_parameters
import numpy as np

from sklearn.model_selection import KFold
from CDmetrics.nn import NN
from tensorflow.keras.utils import to_categorical


def compute_metric(data, num_folds, target_column, custom, resources):

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_order = [
        [(i + j) % num_folds for i in range(num_folds)] for j in range(num_folds)
    ]
    fold_index = [test_index for _, test_index in kfold.split(data)]
    difficulity = []
    difficulity_index = []

    for folds in tqdm(fold_order):

        train_data_index = []
        for index in folds[0 : len(fold_order) // 2]:
            train_data_index += list(fold_index[index])

        evaluation_data_index = []
        for index in folds[len(fold_order) // 2 : len(fold_order) - 1]:
            evaluation_data_index += list(fold_index[index])
        test_data_index = fold_index[folds[len(fold_order) - 1]]
        train_data = data.iloc[train_data_index]
        evaluation_data = data.iloc[evaluation_data_index]
        test_data = data.iloc[test_data_index]

        train_data_x = train_data.drop(columns=[target_column], axis=1)
        train_data_y = train_data[target_column]
        evaluation_data_x = evaluation_data.drop(columns=[target_column], axis=1)
        evaluation_data_y = evaluation_data[target_column]

        n_classes = len(set(train_data[target_column]))
        if n_classes > 2:
            train_data_y = to_categorical(train_data_y, num_classes=n_classes)

        best_model_A = NN(
            tune_parameters(NN.tune, train_data, target_column, custom, resources)
        )

        model_A = best_model_A.train(train_data_x, train_data_y)
        best_model_A_predictions = model_A.predict(evaluation_data_x)

        # Provide 0 to incorrect, 1 to correct
        evaluation_data[target_column] = pd.DataFrame(
            (
                (
                    np.argmax(best_model_A_predictions, axis=1)
                    == np.array(evaluation_data_y)
                ).astype(int)
            )
            * 1,
            index=evaluation_data_x.index,
        )
        best_model_B = NN(
            tune_parameters(NN.tune, evaluation_data, target_column, custom, resources)
        )
        model_B = best_model_B.train(
            evaluation_data.drop(columns=[target_column], axis=1),
            evaluation_data[target_column].values,
        )
        test_data_x = test_data.drop(columns=[target_column], axis=1)
        predicted_difficulty = 1 - model_B.predict(test_data_x, verbose=0).reshape(-1)
        difficulity.extend(predicted_difficulty)
        difficulity_index.extend(test_data_index)
    return pd.DataFrame(difficulity, index=difficulity_index)
