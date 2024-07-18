import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from d_case_difficulty_metrics.api import PATH


class NN:

    def __init__(self, params,input_size, num_classes) -> None:

        

        self.model = tf.keras.models.Sequential()
    # First hidden layer with input shape
        self.model .add(
            tf.keras.layers.Dense(
                params["hidden_layer_sizes"][0],
                input_shape=(input_size),
                activation=params["activation"],
            )
        )
        # Second hidden layer to number of hidden layers
        for i in range(1, len(params["hidden_layer_sizes"])):
            self.model .add(
                tf.keras.layers.Dense(
                    params["hidden_layer_sizes"][i], activation=params["activation"]
                )
            )
        # Ouput layer
        self.model .add(tf.keras.layers.Dense(num_classes, activation=params["activation"]))
        self.model .compile(
            loss=params["loss_function"],
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learnRate"]),
            metrics=["accuracy"],
        )

    def train(self,params, data_x,data_y):
        """
        """
        self.model.compile(
        loss=params["loss_function"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learnRate"]),
        metrics=["accuracy"],
    )

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
        result = self.model.fit(
            data_x,
            data_x,
            verbose=0,
            validation_split=0.3,
            batch_size=params["batch_size"],
            epochs=100,
            callbacks=[es],
        )
        validation_loss = np.amin(result.history["val_loss"])
        validation_acc = np.amax(result.history["val_accuracy"])

        return {
            "loss": validation_loss,
            "acc": validation_acc,
            "status": STATUS_OK,
            "model": self.model,
            "params": params,
        }