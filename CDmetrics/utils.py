import logging
import ray
from ray import tune, train
import itertools
import warnings
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")


def get_available_resources():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    if not ray.is_initialized():
        ray.init(logging_level=logging.CRITICAL, log_to_driver=False)
    resource = ray.cluster_resources()
    if ray.is_initialized():
        ray.shutdown()
    return {"CPU": resource.get("CPU"), "GPU": resource.get("GPU")}


def tune_parameters(model, data, target_column, custom, resources):
    """ """
    if not ray.is_initialized():
        ray.init(logging_level=logging.CRITICAL, log_to_driver=False)

    X = data.drop(target_column, axis=1)
    Y = data[target_column]

    n_classes = len(set(Y.values))
    if n_classes > 2:
        Y = to_categorical(Y, num_classes=n_classes)

    def generate_nn_architecture(max_layers, max_units):
        neurons = list(range(1, max_layers + 1))
        layer_neuron_orders = [
            combination
            for r in range(1, max_units)
            for combination in itertools.product(neurons, repeat=r)
        ]
        return layer_neuron_orders

    max_layers = custom.get("max_layers", 3)
    max_units = custom.get("max_units", 3)
    learn_rate = custom.get("learnRate", [0.01, 0.03, 0.1])
    batch_size = custom.get("batch_size", [32, 64, 128])
    activation = custom.get("activation", ["relu", "tanh"])
    metric = custom.get("metric", "accuracy")
    validation_split = custom.get("validation_split", 0.3)
    epochs = custom.get("epochs", 100)

    search_space = {
        "learnRate": tune.grid_search(learn_rate),
        "batch_size": tune.grid_search(batch_size),
        "activation": tune.grid_search(activation),
        "nn_architecture": tune.grid_search(
            generate_nn_architecture(max_layers, max_units)
        ),
        "num_classes": n_classes,
        "input_size": len(X.columns),
        "metric": metric,
        "validation_split": validation_split,
        "epochs": epochs,
    }

    trainer_with_resources = tune.with_resources(
        tune.with_parameters(model, data_x=X, data_y=Y), resources=resources
    )
    tuner = tune.Tuner(
        trainer_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
        ),
        run_config=train.RunConfig(verbose=0, log_to_file=False),
    )
    tuning_result = tuner.fit()
    best_config = tuning_result.get_best_result().config

    return best_config
