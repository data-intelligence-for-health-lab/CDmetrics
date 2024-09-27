from ray import air, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import itertools



def tune_parameters(model,data,target_column):
    """
    """
    neurons = [1, 1, 1, 1]
    layer_neuron_orders = [
        combination
        for r in range(1, 2)
        for combination in itertools.product(neurons, repeat=r)
    ]
    algo = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=4)
    search_space = {
    "learnRate": tune.choice( [ 0.1]),
    "batch_size": tune.choice( [128]),
    "activation": tune.choice( ["relu"]),
    "number_of_neurons": tune.choice(layer_neuron_orders),

}
    X = data.drop(target_column,axis=1)
    Y = data[target_column]

    trainer_with_resources = tune.with_resources(
            tune.with_parameters(model, data_x=X, data_y=Y),resources={"CPU":3})
    tuner = tune.Tuner(
            trainer_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig( 
                metric="mean_accuracy",
                mode="max",
            ),
        )
    tuning_result = tuner.fit()
    best_config = tuning_result.get_best_result().config

    return best_config