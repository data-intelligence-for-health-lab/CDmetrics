import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from keras.utils import np_utils

from d_case_difficulty_metrics.src.processing.curr_status import curr_status
from d_case_difficulty_metrics.api import PATH
from nn import NN

import multiprocessing
import sys


# manually generate the different random seed
def random_generater():
    return random.randint(1, 100000)


# Model Complexity (NN)
def nn_model_complexity_multiprocessing(
    X, y, X_test, y_test, processing, number_of_neuron, output
):
    count = 0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=random_generater()
    )

    X_train_processed = processing.fit_transform(X_train)
    X_val_processed = processing.transform(X_val)
    X_test_processed = processing.transform(X_test)

    print("y_train:", y_train)
    num_classes = len(unique_labels(y_train))
    print("num_classes:", num_classes)
    if num_classes > 2:
        y_train_processed = np_utils.to_categorical(y_train, num_classes)
        y_val_processed = np_utils.to_categorical(y_val, num_classes)
        loss_function = "categorical_crossentropy"
        output_activation = "softmax"
    else:
        num_classes = 1
        y_train_processed = y_train
        y_val_processed = y_val
        loss_function = "binary_crossentropy"
        output_activation = "sigmoid"

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(
            number_of_neuron,
            input_shape=(X_train_processed.shape[1],),
            activation="relu",
        )
    )  # Increasing number of neuron
    model =NN(num_classes,)
    result = model.train()
    y_test = np.argmax(y_test)
    y_pred = np.argmax(model.predict(np.array(X_test_processed)), axis=1)

    if np.all(y_pred == y_test):
        count += 1
    else:
        count += 0

    tf.keras.backend.clear_session()
    output.put(count)

