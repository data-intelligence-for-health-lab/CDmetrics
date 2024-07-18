import pandas as pd
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
def create_model_A(params):
    num_classes = len(unique_labels(train_y_model_A))
    if num_classes > 2:
        processed_train_y_model_A = np_utils.to_categorical(
            train_y_model_A, num_classes
        )
        loss_function = "categorical_crossentropy"
        output_activation = "softmax"
    else:
        num_classes = 1
        processed_train_y_model_A = train_y_model_A
        loss_function = "binary_crossentropy"
        output_activation = "sigmoid"
    

    model = NN(params, num_classes,output_activation)
 
    results = model.train(params,)
    return  results

# Model B
def create_model_B(params):
    model = NN(params,1,"sigmoid")
    results = model.train()
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

    # First and Second folds
    fold_data0 = data.loc[fold_index[folds[0]]]
    fold_data1 = data.loc[fold_index[folds[1]]]
    train_data_for_model_A = pd.concat([fold_data0, fold_data1], ignore_index=True)

    # Train data for model A
    global train_x_model_A, train_y_model_A
    train_x_model_A = train_data_for_model_A.drop(columns=[target_column], axis=1)
    train_y_model_A = train_data_for_model_A[target_column].to_numpy()

    # Processing model fitted by first and second fold
    train_x_model_A = processing.fit_transform(train_x_model_A)

    # Hyperparam search space for hyperopt
    search_space = {
        "learnRate": hp.choice("learnRate", [0.01, 0.03, 0.1]),
        "batch_size": scope.int(hp.choice("batch_size", [32, 64, 128])),
        "activation": hp.choice("activation", ["relu", "tanh"]),
        "hidden_layer_sizes": hp.choice("hidden_layer_sizes", layer_neuron_orders),
    }

    trials = Trials()
    fmin(
        fn=create_model_A,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_eval_a,
        trials=trials,
    )

    min_loss_index = np.argmin([r["loss"] for r in trials.results])
    best_model_A = trials.results[min_loss_index]["model"]
    best_params_A = trials.results[min_loss_index]["params"]
    best_acc_A = trials.results[min_loss_index]["acc"]
    best_loss_A = trials.results[min_loss_index]["loss"]
    print("best_acc_A:", best_acc_A, "best_loss_A:", best_loss_A)
    print("best_params_A:", best_params_A)

    # Third and Fourth folds
    fold_data2 = data.loc[fold_index[folds[2]]]
    fold_data3 = data.loc[fold_index[folds[3]]]
    test_data_for_model_A = pd.concat([fold_data2, fold_data3], ignore_index=True)

    # Test data for model_A without target column and answer
    model_A_X_test = test_data_for_model_A.drop(columns=[target_column], axis=1)
    answer = test_data_for_model_A[target_column]

    # Processing model fitted by first and second fold transform the Test data
    model_A_X_test = processing.transform(model_A_X_test)
    model_A_X_test = pd.DataFrame(model_A_X_test)

    # Traind model A make predictions
    model_A_predictions = np.argmax(
        best_model_A.predict(np.array(model_A_X_test)), axis=1
    )

    # Compare between answer and predictions from model A
    id_df = pd.DataFrame({"actual": answer, "predicted": model_A_predictions})
    incorrect = id_df[id_df["actual"] != id_df["predicted"]]

    incorrect_index = []
    incorrect_index = incorrect.index

    # Provide 0 to incorret, 1 to correct using incorrect_index
    correctness = pd.Series(1, index=model_A_X_test.index)
    correctness.loc[incorrect_index] = 0
    correctness_df = correctness.to_frame(name="correctness")

    # Get the original test data from fold 3, fold 4 (Before processing)
    test_data_for_model_A = pd.concat([fold_data2, fold_data3], ignore_index=True)
    model_A_X_test = test_data_for_model_A.drop(columns=[target_column], axis=1)

    # Processing model fitted by thrid and fourth folds
    model_A_X_test = processing.fit_transform(model_A_X_test)

    # Fold 3, 4 data with correctness
    global model_B_train_x, model_B_train_y
    model_B_train_x = model_A_X_test
    model_B_train_y = correctness_df.to_numpy()

    trials = Trials()
    fmin(
        fn=create_model_B,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_eval_b,
        trials=trials,
    )

    min_loss_index = np.argmin([r["loss"] for r in trials.results])
    best_model_B = trials.results[min_loss_index]["model"]
    best_params_B = trials.results[min_loss_index]["params"]
    best_acc_B = trials.results[min_loss_index]["acc"]
    best_loss_B = trials.results[min_loss_index]["loss"]
    print("best_acc_B:", best_acc_B, "best_loss_B:", best_loss_B)
    print("best_params_B:", best_params_B)

    # Predict fold 5 correctness probability
    difficulty_data_for_model_B = data.loc[fold_index[folds[4]]]
    difficulty_x = difficulty_data_for_model_B.drop(columns=[target_column], axis=1)

    # Processing model fitted by third and fourth fold transform the fold 5
    difficulty_x = processing.transform(difficulty_x)

    # By doing 1- Difficult case is closer to 1, Easy case is closer to 0
    predicted_difficulty = 1 - best_model_B.predict(difficulty_x)
    predicted_difficulty = predicted_difficulty[:, 0].tolist()

    return (difficulty_data_for_model_B, predicted_difficulty)

