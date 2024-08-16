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

    train_data_index = []
    for index in  folds[0:len(folds//2)]:
        train_data_index += list(fold_index[index])
    
    evaluation_data_index = []
    for index in  folds[len(folds//2):len(folds)-1]:
        train_data_model_B_indexex += list(fold_index[index])

    train_data = data.iloc[train_data_index]
    evaluation_data = data.iloc[evaluation_data_index]
    test_data = data.iloc[len(folds)]

   
    best_model_A = tune_parameters(train_data)


    # Test data for model_A without target column and answer
    evaluation_data_x = evaluation_data.drop(columns=[target_column], axis=1)
    evaluation_data_y = evaluation_data[target_column]


    # Traind model A make predictions
    best_model_A_predictions = best_model_A.predict(evaluation_data_x)
    

    # Compare between answer and predictions from model A
    
    correctness_df = pd.DataFrame((np.array(best_model_A_predictions) != np.array(evaluation_data_y))*1,index=evaluation_data_x.index)


    # Get the original test data from fold 3, fold 4 (Before processing)

    best_model_B = tune_parameters(evaluation_data_x,correctness_df)

    # Predict fold 5 correctness probability

    test_data_x = test_data.drop(columns=[target_column], axis=1)

    # By doing 1- Difficult case is closer to 1, Easy case is closer to 0
    predicted_difficulty = 1 - best_model_B.predict(test_data_x)

    return  predicted_difficulty

