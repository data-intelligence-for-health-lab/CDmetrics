import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from d_case_difficulty_metrics.api import PATH


class NN:
    def __init__(self, params) -> None:
        self.num_classes = params["num_classes"]
        self.model = tf.keras.models.Sequential()
        self.batch_size = params["batch_size"]
        self.params = params
        # First hidden layer with input shape
        self.model.add(
            tf.keras.layers.Dense(
                params["number_of_neuron"][0],
                input_shape=(params["input_size"]),
                activation=params["activation"],
            )
        )
        # Second hidden layer to number of hidden layers
        for i in range(1, len(params["hidden_layer_sizes"])):
            self.model.add(
                tf.keras.layers.Dense(
                    params["number_of_neuron"][i], activation=params["activation"]
                )
            )

        if self.num_classes > 2:
            self.loss_function = "categorical_crossentropy"
            self.output_activation = "softmax"
        else:
            self.num_classes = 1
            self.loss_function = "binary_crossentropy"
            self.output_activation = "sigmoid"

        # Ouput layer
        self.model.add(
            tf.keras.layers.Dense(self.num_classes, activation=self.output_activation)
        )
        self.model.compile(
            loss=self.loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learnRate"]),
            metrics=["accuracy"],
        )

    def tune(config, data_x, data_y):
        """ """
        model = NN(config)
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
        model.fit(
            data_x,
            data_y,
            verbose=0,
            validation_split=0.3,
            batch_size=config["batch_size"],
            epochs=100,
            callbacks=[
                es,
                ReportCheckpointCallback(metrics={"mean_accuracy": "accuracy"}),
            ],
        )

    def train(self, number_of_neuron, data_x, data_y):
        """ """
        self.params["number_of_neuron"] = number_of_neuron
        model = self.model(self.params)
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
        model.fit(
            data_x,
            data_y,
            verbose=0,
            validation_split=0.3,
            batch_size=32,
            epochs=100,
            callbacks=[
                es,
                ReportCheckpointCallback(metrics={"mean_accuracy": "accuracy"}),
            ],
        )

    def predict():
        """ """
