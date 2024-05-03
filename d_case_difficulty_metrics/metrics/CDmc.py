import os

import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils

from d_case_difficulty_metrics.compute_case_difficulty.processing import (
    check_curr_status,
)
from d_case_difficulty_metrics.api import PATH

import multiprocessing

import sys


# manually generate the different random seed
def random_generater():
    return random.randint(1, 100000)


# Model Complexity (NN)
def nn_model_complexity_multiprocessing(
    X, y, X_test, y_test, processing, number_of_neuron
):
    count = 0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=random_generater()
    )

    X_train_processed = processing.fit_transform(X_train)
    X_val_processed = processing.transform(X_val)
    X_test_processed = processing.transform(X_test)

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
    model.add(
        tf.keras.layers.Dense(
            number_of_neuron,
            input_shape=(X_train_processed.shape[1],),
            activation="relu",
        )
    )  # Increasing number of neuron
    model.add(
        tf.keras.layers.Dense(num_classes, activation=output_activation)
    )  # output layer
    model.compile(loss=loss_function, optimizer="adam", metrics=["accuracy"])

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=0, patience=30
    )

    model.fit(
        X_train_processed,
        y_train_processed,
        validation_data=(X_val_processed, y_val_processed),
        batch_size=32,
        epochs=100,
        verbose=0,
        callbacks=[es],
    )

    y_test = np.argmax(y_test)
    y_pred = np.argmax(model.predict(np.array(X_test_processed)), axis=1)

    if np.all(y_pred == y_test):
        count += 1
    else:
        count += 0

    tf.keras.backend.clear_session()
    output.put(count)


rows_value = []
output = multiprocessing.Queue()


def CDmc_run(file_name, data, processing, target_column, number_of_NNs):
    n_samples = len(data)
    starting, curr_df = check_curr_status(n_samples, PATH + file_name)

    rows = []
    # CDmc_set
    for index in range(starting, 2):  # number of index to check
        print("\nindex:", index)

        X_overall = data.drop(columns=[target_column], axis=1)
        y_overall = data[target_column]

        # One sample left out
        X_test = X_overall.iloc[
            [index]
        ]  # the test case that want to check the difficulty
        y_test = y_overall[index]

        X = X_overall.drop(index=[index])  # X,y the dataset wilthout the test case
        y = y_overall.drop(index=[index])

        correct_count = 0
        processes = []
        for _ in range(0, number_of_NNs):  # How many NN to generate
            p = multiprocessing.Process(
                target=nn_model_complexity_multiprocessing,
                args=(X, y, X_test, y_test, processing, 1),
            )
            p.start()
            processes.append(p)

        try:
            for process in processes:
                process.join()

            # Get process results from the output queue
            correct_count = [output.get() for p in processes]
            # print("correct_count:",correct_count)
            # print("correct number:",sum(correct_count))

            rows = (
                [index]
                + [X_test[column][index] for column in X_test.columns]
                + [y_test]
                + int(1)
                + [sum(correct_count)]
            )
            print("rows:", rows)
            rows_value.append(rows)
            results_df = pd.DataFrame(rows_value)
            # print("results:",results)

            # Append the dictionary as a new row in the DataFrame
            results_df = results_df.append(rows, ignore_index=True)

        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
                process.join()
                print("Keyboard error occurred")
            results_df.to_excel(PATH + "interrupted_" + file_name, index=False)
            sys.exit(0)

        except Exception as e:
            print(f"Error: {e}")
            print("An error occurred")
            results_df.to_excel(PATH + "error_" + file_name, index=False)

        finally:
            # Save the DataFrame to Excel once after the loop completes
            curr_df = pd.concat([curr_df, results_df], ignore_index=True)
            curr_df.to_excel(PATH + file_name, index=False)

    # Adding column_names to excel file
    if len(curr_df) == n_samples:
        curr_df.columns = (
            ["index"] + X.columns.to_list() + ["y", "number_of_neuron", "correct_count"]
        )
        curr_df.to_excel(file_name, index=False)
    else:
        sys.exit(0)

    # CDmc_add
    while True:
        MNN = round(len(data) * 0.01)
        updated_rows = []

        one_neuron_file = pd.read_excel(file_name)

        X_all = one_neuron_file.drop(
            columns=["index", "y", "number_of_neuron", "correct_count"]
        )
        y_all = one_neuron_file["y"]

        # Check the index that needs to be repeated
        repeat_index_count_view = one_neuron_file.loc[
            (one_neuron_file["number_of_neuron"] < MNN)
            & (one_neuron_file["correct_count"] < number_of_NNs * 0.9),
            ["index", "number_of_neuron"],
        ]

        print(
            "The total number of indexes needed to repeat:",
            len(repeat_index_count_view),
        )
        print(repeat_index_count_view, "\n")

        # Check the index that needs to be repeated with the certain number of neuron
        number_of_neuron = repeat_index_count_view["number_of_neuron"].min()
        repeat_index_neuron_count_view = repeat_index_count_view.loc[
            (one_neuron_file["number_of_neuron"] <= number_of_neuron),
            ["index", "number_of_neuron"],
        ]

        print(
            "The number of indexes left (Neuron perspective):",
            len(repeat_index_neuron_count_view),
        )
        print(repeat_index_neuron_count_view, "\n")

        number_of_neuron += 1
        print("Number of Neuron Used:", number_of_neuron)

        if len(repeat_index_neuron_count_view) == 0:
            print("Nothing to repeat")
            sys.exit(0)

        else:
            for i in repeat_index_neuron_count_view["index"].values.tolist():

                X_test = X_all.iloc[
                    [i]
                ]  # the test case that want to check the difficulty
                y_test = y_all[i]

                X = X_all.drop(index=[i])  # X,y the dataset wilthout the test case
                y = y_all.drop(index=[i])

                processes = []
                for NNs in range(number_of_NNs):  # How many NN to generate
                    p = multiprocessing.Process(
                        target=nn_model_complexity_multiprocessing,
                        args=(X, y, X_test, y_test, number_of_neuron),
                    )
                    p.start()
                    processes.append(p)

                try:
                    for process in processes:
                        process.join()
                    # Get process results from the output queue
                    correct_count = [output.get() for p in processes]
                    print("index", i, "correct number:", sum(correct_count))

                    updated_rows.append(
                        [i]
                        + [X_test[column][i] for column in X_test.columns]
                        + [y_test]
                        + [number_of_neuron]
                        + [sum(correct_count)]
                    )
                    # print(updated_rows)

                    score = pd.DataFrame(updated_rows, columns=one_neuron_file.columns)
                    result = (
                        pd.concat([one_neuron_file, score])
                        .drop_duplicates(["index"], keep="last")
                        .sort_values("index")
                    )
                    result = result.reset_index(drop=True)
                    # print(result)

                    writer = pd.ExcelWriter(file_name)
                    result.to_excel(writer, index=False)
                    writer.close()

                except KeyboardInterrupt:
                    print("parent received ctrl-c")
                    for process in processes:
                        process.terminate()
                        process.join()
                    sys.exit("parent received ctrl-c.")
