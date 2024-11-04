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


def tune_parameters(model, data, target_column, max_layers, max_units, resources):
    """ """
    if not ray.is_initialized():
        ray.init(logging_level=logging.CRITICAL, log_to_driver=False)

    neurons = list(range(1, max_layers + 1))
    layer_neuron_orders = [
        combination
        for r in range(1, max_units)
        for combination in itertools.product(neurons, repeat=r)
    ]
    X = data.drop(target_column, axis=1)
    Y = data[target_column]

    n_classes = len(set(Y.values))
    if n_classes > 2:
        Y = to_categorical(Y, num_classes=n_classes)   

    search_space = {
        "learnRate": tune.grid_search([0.01, 0.03, 0.1]),
        "batch_size": tune.grid_search([32, 64, 128]),
        "activation": tune.grid_search(["relu", "tanh"]),
        "number_of_neurons": tune.grid_search(layer_neuron_orders),
        "num_classes": n_classes,
        "input_size": len(X.columns),
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
