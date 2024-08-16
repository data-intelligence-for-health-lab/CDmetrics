from ray import air, tune





def tune_parameters(search_space,model,data):
    """
    """
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