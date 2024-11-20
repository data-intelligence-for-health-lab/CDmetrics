
import logging
import ray
from ray import tune,train
from ray.train import Result, RunConfig, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
import itertools
import warnings
warnings.filterwarnings('ignore')

def get_available_resources():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    if not ray.is_initialized():
        ray.init(logging_level=logging.CRITICAL,log_to_driver=False)
    resource = ray.cluster_resources()
    if ray.is_initialized():
        ray.shutdown()
    return {'CPU':resource.get('CPU'),'GPU':resource.get('GPU')}

def tune_parameters(model,data,target_column,max_layers,max_units, resources):
    """
    """
    if not ray.is_initialized():
        ray.init(logging_level=logging.CRITICAL,log_to_driver=False)

    neurons = list(range(1, max_layers+1))
    layer_neuron_orders = [
        combination
        for r in range(1, max_units)
        for combination in itertools.product(neurons, repeat=r)
    ]
    X = data.drop(target_column,axis=1)
    Y = data[target_column]
    search_space = {
    "learnRate": tune.choice( [ 0.1]),
    "batch_size": tune.choice( [128]),
    "activation": tune.choice( ["relu"]),
    "number_of_neurons": tune.choice(layer_neuron_orders),
    "num_classes":   len(set(Y.values)),
    "input_size" : len(X.columns)

}

    trainer_with_resources = tune.with_resources(
            tune.with_parameters(model, data_x=X, data_y=Y),resources=resources)
    tuner = tune.Tuner(
        
            trainer_with_resources,

            param_space=search_space,
            tune_config=tune.TuneConfig( 
                metric="mean_accuracy",
                mode="max",
            ),
            run_config=train.RunConfig(verbose=0,log_to_file=False),

        )
    tuning_result = tuner.fit()
    best_config = tuning_result.get_best_result().config

    return best_config


def train_model(
    model,
    num_workers: int = 2,
    use_gpu: bool = False,
    epochs: int = 4,
    storage_path: str = None,
) -> Result:
    config = {"lr": 1e-3, "batch_size": 64, "epochs": epochs}
    trainer = TensorflowTrainer(
        train_loop_per_worker=model,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(storage_path=storage_path),
    )
    results = trainer.fit()
    return results


