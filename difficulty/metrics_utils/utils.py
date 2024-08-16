from ray import air, tune





def tune_parameters(search_space,model,data):
    """
    """
    search_space = {
    "learnRate": hp.choice("learnRate", [0.01, 0.03, 0.1]),
    "batch_size": scope.int(hp.choice("batch_size", [32, 64, 128])),
    "activation": hp.choice("activation", ["relu", "tanh"]),
    "hidden_layer_sizes": hp.choice("hidden_layer_sizes", layer_neuron_orders),
}


    trainer_with_resources = tune.with_resources(
            tune.with_parameters(model, data_frame=data),)
    tuner = tune.Tuner(
            trainer_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=20,
                metric=model.get("metrics").get("name"),
                mode=model.get("metrics").get("mode", "min"),
            ),
        )
    tuning_result = tuner.fit()
    best_config = tuning_result.get_best_result().config

    return best_config