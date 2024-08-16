import pandas as pd
from difficulty.metrics_utils.utils import tune_parameters
import numpy as np
import itertools

from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels
from  nn import NN

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from d_case_difficulty_metrics.api import PATH


# Model A
def fit_model_A(params,data):
    data_x, data_y = data
    model = NN(params)
    results = model.train(params,data_x, data_y)
    return  results

# Model B
def fit_model_B(params,data):
    data_x, data_y = data
    model = NN(params)
    results = model.train(data_x,data_y)
    return results

def CDdm_main(data, folds, max_eval_a, max_eval_b, processing, target_column):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_index = [test_index for _, test_index in kfold.split(data)]

    # Generate combinations of layers and neurons
    neurons = [5, 10, 15, 20]
    layer_neuron_orders = [
        combination
        for r in range(1, 4)
        for combination in itertools.product(neurons, repeat=r)
    ]

    train_data_model_A_indexex = []
    for index in  folds[0:len(folds//2)]:
        train_data_model_A_indexex += list(fold_index[index])

    # train_data_for_model_A = pd.concat([fold_data0, fold_data1], ignore_index=True)
    train_data_for_model_A = data.iloc[train_data_model_A_indexex]
    # Train data for model A

    train_data_for_model_A_x = train_data_for_model_A.drop(columns=[target_column], axis=1)
    train_data_for_model_A_y = train_data_for_model_A[target_column].to_numpy()

    # Hyperparam search space for hyperopt
    search_space = {
        "learnRate": hp.choice("learnRate", [0.01, 0.03, 0.1]),
        "batch_size": scope.int(hp.choice("batch_size", [32, 64, 128])),
        "activation": hp.choice("activation", ["relu", "tanh"]),
        "hidden_layer_sizes": hp.choice("hidden_layer_sizes", layer_neuron_orders),
    }


    # Third and Fourth folds
    best_model_A = tune_parameters()

    test_data_model_A_indexex = []
    for index in  folds[len(folds//2):len(folds)-1]:
        train_data_model_B_indexex += list(fold_index[index])

    
    test_data_for_model_A = data.iloc[test_data_model_A_indexex]

    # Test data for model_A without target column and answer
    test_data_for_model_A_x = test_data_for_model_A.drop(columns=[target_column], axis=1)
    test_data_for_model_A_y = test_data_for_model_A[target_column]


    # Traind model A make predictions
    best_model_A_predictions = np.argmax(
        best_model_A.predict(np.array(test_data_for_model_A_x)), axis=1
    )

    # Compare between answer and predictions from model A
    id_df = pd.DataFrame({"actual": test_data_for_model_A_y, "predicted": best_model_A_predictions})
    incorrect = id_df[id_df["actual"] != id_df["predicted"]]

    incorrect_index = []
    incorrect_index = incorrect.index

    # Provide 0 to incorret, 1 to correct using incorrect_index
    correctness = pd.Series(1, index=test_data_for_model_A_x.index)
    correctness.loc[incorrect_index] = 0
    correctness_df = correctness.to_frame(name="correctness")

    # Get the original test data from fold 3, fold 4 (Before processing)

    best_model_B = tune_parameters()

    # Predict fold 5 correctness probability
    difficulty_data_for_model_B = data.loc[fold_index[folds[4]]]
    difficulty_x = difficulty_data_for_model_B.drop(columns=[target_column], axis=1)

    # Processing model fitted by third and fourth fold transform the fold 5
    difficulty_x = processing.transform(difficulty_x)

    # By doing 1- Difficult case is closer to 1, Easy case is closer to 0
    predicted_difficulty = 1 - best_model_B.predict(difficulty_x)
    predicted_difficulty = predicted_difficulty[:, 0].tolist()

    return (difficulty_data_for_model_B, predicted_difficulty)

