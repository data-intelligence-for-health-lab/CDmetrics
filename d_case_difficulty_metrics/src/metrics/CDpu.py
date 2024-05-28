import pandas as pd
import numpy as np
import itertools
import random
import sys

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from multiprocessing import Pool, Manager
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import ray


from functools import partial

from d_case_difficulty_metrics.api import PATH
from d_case_difficulty_metrics.src.processing.curr_status import curr_status


# manually generate the different random seed
def random_generater():
    return random.randint(1, 100000)


def objective(X_without_test, y_without_test, processing, config):
    X_train, X_val, y_train, y_val = train_test_split(
        X_without_test, y_without_test, test_size=0.3, random_state=random_generater()
    )

    X_train_processed = processing.fit_transform(X_train)
    X_val_processed = processing.transform(X_val)

    num_classes = len(unique_labels(y_train))
    if num_classes > 2:
        y_train_processed = np_utils.to_categorical(y_train, num_classes)
        y_val_processed = np_utils.to_categorical(y_val, num_classes)
        loss_function = "categorical_crossentropy"
        output_activation = "softmax"
    else:
        num_classes = 1
        y_train_processed = y_train
        y_val_processed = y_val
        loss_function = "binary_crossentropy"
        output_activation = "sigmoid"

    model = tf.keras.models.Sequential()

    # First hidden layer with input shape
    model.add(
        tf.keras.layers.Dense(
            config["hidden_layer_sizes"][0],
            input_shape=(X_train_processed.shape[1],),
            activation=config["activation"],
        )
    )

    # Ssecond hidden layer to number of hidden layers
    for i in range(1, len(config["hidden_layer_sizes"])):
        model.add(
            tf.keras.layers.Dense(
                config["hidden_layer_sizes"][i], activation=config["activation"]
            )
        )

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation=output_activation))
    model.compile(
        loss=loss_function,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learnRate"]),
        metrics=["accuracy"],
    )

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=30)
    result = model.fit(
        X_train_processed,
        y_train_processed,
        validation_data=(X_val_processed, y_val_processed),
        verbose=0,
        batch_size=config["batch_size"],
        epochs=100,
        callbacks=[es],
    )

    val_loss = result.history["val_loss"][-1]
    return {"val_loss": val_loss}


def hyperparm_searching(hyper_file_name, data, processing, target_column):
    n_samples = len(data)
    starting, curr_df = curr_status(n_samples, PATH + hyper_file_name)

    # Generate combinations of layers and neurons
    neurons = [5, 10, 15, 20]
    layer_neuron_orders = [
        combination
        for r in range(1, 4)
        for combination in itertools.product(neurons, repeat=r)
    ]

    # Initialize an empty DataFrame
    results_df = pd.DataFrame(
        columns=["Index", "LearnRate", "BatchSize", "Activation", "HiddenLayerSizes"]
    )

    try:
        for index in range(starting, len(data)):
            X_overall = data.drop(columns=[target_column], axis=1)
            y_overall = data[target_column]

            X_without_test = X_overall.drop(index=[index])
            y_without_test = y_overall.drop(index=[index])

            configured_objective = partial(
                objective, X_without_test, y_without_test, processing
            )

            search_space = {
                "learnRate": tune.choice([0.01, 0.03, 0.1]),
                "batch_size": tune.choice([32, 64, 128]),
                "activation": tune.choice(["relu", "tanh"]),
                "hidden_layer_sizes": tune.choice(layer_neuron_orders),
            }

            tuner = tune.Tuner(
                configured_objective,
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    num_samples=3,
                    search_alg=HyperOptSearch(),
                ),
                run_config=ray.air.config.RunConfig(verbose=0),
                param_space=search_space,
            )

            hyper_param = tuner.fit().get_best_result().config
            print("Best hyperparameters found were: ", hyper_param)

            rows = {
                "Index": index,
                "LearnRate": hyper_param["learnRate"],
                "BatchSize": hyper_param["batch_size"],
                "Activation": hyper_param["activation"],
                "HiddenLayerSizes": str(hyper_param["hidden_layer_sizes"]),
            }

            # Append the dictionary as a new row in the DataFrame
            results_df = results_df.append(rows, ignore_index=True)

        # Save the DataFrame to Excel once after the loop completes
        curr_df = pd.concat([curr_df, results_df], ignore_index=True)
        curr_df.to_excel(PATH + hyper_file_name, index=False)

    except KeyboardInterrupt:
        print("Keyboard error occurred")
        results_df.to_excel(PATH + "interrupted_" + hyper_file_name, index=False)
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred")
        results_df.to_excel(PATH + "error_" + hyper_file_name, index=False)

    finally:
        ray.shutdown()


# Model Complexity (NN)
def nn_model_complexity_multiprocessing(
    X_train, X_val, y_train, y_val, X_test, processing, best_params
):
    num_classes = len(unique_labels(y_train))
    if num_classes > 2:
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_val = np_utils.to_categorical(y_val, num_classes)
        loss_function = "categorical_crossentropy"
        output_activation = "softmax"
    else:
        num_classes = 1
        loss_function = "binary_crossentropy"
        output_activation = "sigmoid"

    X_train_processed = processing.fit_transform(X_train)
    X_val_processed = processing.transform(X_val)
    X_test_processed = processing.transform(X_test)

    model = tf.keras.models.Sequential()
    # First hidden layer with input shape
    model.add(
        tf.keras.layers.Dense(
            best_params["hidden_layer_sizes"][0],
            input_shape=(X_train_processed.shape[1],),
            activation=best_params["activation"],
        )
    )
    for i in range(1, len(best_params["hidden_layer_sizes"])):
        # from second hidden layer to number of hidden layers
        model.add(
            tf.keras.layers.Dense(
                best_params["hidden_layer_sizes"][i],
                activation=best_params["activation"],
            )
        )
    # Ouput layer
    model.add(tf.keras.layers.Dense(num_classes, activation=output_activation))
    model.compile(
        loss=loss_function,
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learnRate"]),
        metrics=["accuracy"],
    )
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=30)
    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        verbose=0,
        batch_size=best_params["batch_size"],
        epochs=100,
        callbacks=[es],
    )
    difficulty = model.predict(X_test_processed, verbose=0)
    return difficulty


def CDpu_run(
    file_name,
    hyper_file_name,
    data,
    processing,
    target_column,
    number_of_predictions,
    number_of_cpu,
):
    hyper_param_df = pd.read_excel(
        hyper_file_name,
        index_col=None,
        header=None,
        names=["index", "learnRate", "batch_size", "activation", "hidden_layer_sizes"],
    )

    hyper_param_df["hidden_layer_sizes"] = hyper_param_df["hidden_layer_sizes"].apply(
        lambda x: eval(x)
    )

    # Check hyperparam file prepared properly
    if len(hyper_param_df) != len(data):
        sys.exit("Error: Number of hyper_param is not enough.")

    n_samples = len(data)
    starting = curr_status(n_samples, file_name)

    try:
        all_row_values = []
        for index in range(starting, n_samples):
            X_overall = data.drop(columns=[target_column], axis=1)
            y_overall = data[target_column]

            X_test = X_overall.iloc[[index]]  # test case want to check the difficulty
            y_test = y_overall[index]

            X_without_test = X_overall.drop(index=[index])
            y_without_test = y_overall.drop(index=[index])
            param = hyper_param_df.iloc[index]

            X_train_dataset = []
            X_val_dataset = []
            y_train_dataset = []
            y_val_dataset = []

            for _ in range(number_of_predictions):
                X_train, X_val, y_train, y_val = train_test_split(
                    X_without_test,
                    y_without_test,
                    test_size=0.3,
                    random_state=random_generater(),
                )
                X_train_dataset.append(X_train)
                X_val_dataset.append(X_val)
                y_train_dataset.append(y_train)
                y_val_dataset.append(y_val)

            manager = Manager()
            predicted_probabilities = manager.list()

            def collect_result(result):
                predicted_probabilities.append(result)

            # Create a list of argument tuples
            arg_list = [
                (
                    X_train_dataset[mm],
                    X_val_dataset[mm],
                    y_train_dataset[mm],
                    y_val_dataset[mm],
                    X_test,
                    processing,
                    param,
                )
                for mm in range(number_of_predictions)
            ]

            try:
                pool = Pool(processes=number_of_cpu)
                for args in arg_list:
                    pool.apply_async(
                        nn_model_complexity_multiprocessing,
                        args,
                        callback=collect_result,
                    )
                pool.close()
                pool.join()

            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Terminating...")
                pool.terminate()
                pool.join()

            finally:
                if len(predicted_probabilities) != number_of_predictions:
                    results_df = pd.DataFrame(all_row_values)
                    results_df.to_excel(PATH + "unmatching_" + file_name, index=False)
                    sys.exit("Error: Number of predicted_probabilities does not match.")

                else:
                    # index, X_test, y_test, predicted_probabilities
                    # binary class: single prediction, multiclass: each class prediction
                    result_array = np.concatenate(
                        [arr.flatten() for arr in predicted_probabilities]
                    ).tolist()

                    row_values = (
                        [index]
                        + [X_test[column][index] for column in X_test.columns]
                        + [y_test]
                        + result_array
                    )
                    all_row_values.append(row_values)
                    print("all_row_values:", all_row_values)

        results_df = pd.DataFrame(all_row_values)
        results_df.to_excel(PATH + file_name, index=False)

        return results_df

    except KeyboardInterrupt:
        print("Keyboard error occurred")
        results_df = pd.DataFrame(all_row_values)
        results_df.to_excel(PATH + "interrupted_" + file_name, index=False)
        sys.exit(0)

    except Exception:
        print("An error occurred")
        results_df = pd.DataFrame(all_row_values)
        results_df.to_excel(PATH + "error_" + file_name, index=False)
        sys.exit(0)
