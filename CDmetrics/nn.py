from ray.air.integrations.keras import ReportCheckpointCallback

import tensorflow as tf
from keras.callbacks import EarlyStopping


class NN:
    def __init__(self, params) -> None:
        self.num_classes = params["num_classes"]

        self.batch_size = params["batch_size"]
        self.validation_split = params["validation_split"]
        self.epochs = params["epochs"]
        self.metric = params["metric"]
        self.params = params

        # First hidden layer with input shape
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Dense(
                params["nn_architecture"][0],
                input_shape=(params["input_size"],),
                activation=params["activation"],
            )
        )
        # Second hidden layer to number of hidden layers
        for i in range(1, len(params["nn_architecture"])):
            self.model.add(
                tf.keras.layers.Dense(
                    params["nn_architecture"][i], activation=params["activation"]
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
            metrics=[params["metric"]],
        )

    def tune(config, data_x, data_y):
        """ """
        network = NN(config)
        model = network.model
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
        model.fit(
            data_x,
            data_y,
            verbose=0,
            validation_split=config["validation_split"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=[
                es,
                ReportCheckpointCallback(
                    metrics={f"mean_{config['metric']}": config["metric"]}
                ),
            ],
        )

    def train(self, data_x, data_y):
        """ """
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
        self.model.fit(
            data_x,
            data_y,
            verbose=0,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                es,
                ReportCheckpointCallback(metrics={f"mean_{self.metric}": self.metric}),
            ],
        )

        return self.model
