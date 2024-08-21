from ray import air, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from ray.tune.schedulers import AsyncHyperBandScheduler
import itertools



def tune_parameters(model,data,target_column):
    """
    """
    sched = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=400, grace_period=20)
    neurons = [5, 10, 15, 20]
    layer_neuron_orders = [
        combination
        for r in range(1, 4)
        for combination in itertools.product(neurons, repeat=r)
    ]
    algo = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=4)
    search_space = {
    "learnRate": tune.choice("learnRate", [0.01, 0.03, 0.1]),
    "batch_size": tune.choice("batch_size", [32, 64, 128]),
    "activation": tune.choice("activation", ["relu", "tanh"]),
    "hidden_layer_sizes": tune.choice("hidden_layer_sizes", layer_neuron_orders),
}


    trainer_with_resources = tune.with_resources(
            tune.with_parameters(model, data_frame=data),)
    tuner = tune.Tuner(
            trainer_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=1000,
                search_alg=algo,
                metric="mean_loss",
                scheduler=sched,
                mode="min",
            ),
        )
    tuning_result = tuner.fit()
    best_config = tuning_result.get_best_result().config

    return best_config