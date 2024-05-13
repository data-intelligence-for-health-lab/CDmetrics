import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from keras.utils import np_utils

from d_case_difficulty_metrics.src.processing.curr_status import curr_status
from d_case_difficulty_metrics.api import PATH

import multiprocessing
import sys


# manually generate the different random seed
def random_generater():
    return random.randint(1, 100000)


# Model Complexity (NN)
def nn_model_complexity_multiprocessing(
    X, y, X_test, y_test, processing, number_of_neuron, output
):
    count = 0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=random_generater()
    )

    X_train_processed = processing.fit_transform(X_train)
    X_val_processed = processing.transform(X_val)
    X_test_processed = processing.transform(X_test)

    print("y_train:", y_train)
    num_classes = len(unique_labels(y_train))
    print("num_classes:", num_classes)
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

    # TODO change epochs
    model.fit(
        X_train_processed,
        y_train_processed,
        validation_data=(X_val_processed, y_val_processed),
        batch_size=32,
        epochs=5,
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


def CDmc_run(file_name, data, processing, target_column, number_of_NNs):
    output = multiprocessing.Queue()
    n_samples = len(data)
    print("PATH + file_name:", PATH + file_name)
    starting, curr_df = curr_status(n_samples, PATH + file_name)

    rows_value = []
    rows = []
    # CDmc_set

    try:
        # TODO change 3 to n_samples
        for index in range(starting, 3):
            print("\nindex:", index)

            X_overall = data.drop(columns=[target_column], axis=1)
            y_overall = data[target_column]

            # the test case that want to check the difficulty
            X_test = X_overall.iloc[[index]]
            y_test = y_overall[index]

            X = X_overall.drop(index=[index])  # X,y the dataset wilthout the test case
            y = y_overall.drop(index=[index])

            processes = []

            for _ in range(0, number_of_NNs):  # How many NN to generate
                p = multiprocessing.Process(
                    target=nn_model_complexity_multiprocessing,
                    args=(X, y, X_test, y_test, processing, 1, output),
                )
                p.start()
                processes.append(p)

            for process in processes:
                process.join()

            correct_count = []
            while not output.empty():
                correct_count.append(output.get())

            print("correct_count:", correct_count)
            print("correct_count:", sum(correct_count))

            rows = (
                [index]
                + [X_test[column][index] for column in X_test.columns]
                + [y_test]
                + [int(1)]
                + [sum(correct_count)]
            )
            print("rows:", rows)
            rows_value.append(rows)

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()
            print("Keyboard error occurred")

        results_df = pd.DataFrame(rows_value)
        results_df.to_excel(PATH + "interrupted_" + file_name, index=False)
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred")
        results_df = pd.DataFrame(rows_value)
        results_df.to_excel(PATH + "error_" + file_name, index=False)

    finally:
        results_df = pd.DataFrame(rows_value)
        curr_df = pd.concat([curr_df, results_df], ignore_index=True)

        # Adding column_names when the set is done
        # TODO change 3 to n_samples
        if len(curr_df) == 3:
            curr_df.columns = (
                ["index"]
                + data.columns.to_list()
                + ["number_of_neuron", "correct_count"]
            )
            curr_df.to_excel(PATH + file_name, index=False)
        else:
            curr_df.to_excel(PATH + file_name, index=False)

    # CDmc_add

    while True:
        # TODO change to MNN = round(len(data) * 0.01)
        MNN = 3
        one_neuron_file = pd.read_excel(PATH + file_name)

        # Check the index that needs to be repeated
        repeat_index_count_view = one_neuron_file.loc[
            (one_neuron_file["number_of_neuron"] < MNN)
            & (one_neuron_file["correct_count"] < number_of_NNs * 0.9),
            ["index", "number_of_neuron"],
        ]

        print(
            "The total number data need to be repeated:",
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
            "The number of indexes left middle of the running:",
            len(repeat_index_neuron_count_view),
        )
        print(repeat_index_neuron_count_view, "\n")

        number_of_neuron += 1
        print("Number of Neuron Used:", number_of_neuron)

        if len(repeat_index_neuron_count_view) == 0:
            print("Nothing to repeat")
            sys.exit(0)

        else:
            # Drop index, y, number_of_neuron, correct_count
            X_all = one_neuron_file.iloc[:, 1:-3]
            y_all = one_neuron_file.iloc[:, -3]

            print("y_all:", y_all)

            for repeat_index in repeat_index_neuron_count_view["index"].values.tolist():

                # the test case that want to check the difficulty
                X_test = X_all.iloc[[repeat_index]]
                y_test = y_all.iloc[repeat_index]

                print("y_test_2:", y_test)

                # X,y the dataset wilthout the test case
                X = X_all.drop(index=[repeat_index])
                y = y_all.drop(index=[repeat_index])

                print("y3:", y)

                try:
                    rows_value = []
                    rows = []
                    processes = []
                    for NNs in range(number_of_NNs):  # How many NN to generate
                        p = multiprocessing.Process(
                            target=nn_model_complexity_multiprocessing,
                            args=(X, y, X_test, y_test, processing, NNs, output),
                        )
                        p.start()
                        processes.append(p)

                    for process in processes:
                        process.join()
                    # Get process results from the output queue

                    correct_count = []
                    while not output.empty():
                        correct_count.append(output.get())

                    print(
                        "repeat_index",
                        repeat_index,
                        "correct number:",
                        sum(correct_count),
                    )

                    rows = [
                        [repeat_index]
                        + [X_test[column][repeat_index] for column in X_test.columns]
                        + [y_test]
                        + [number_of_neuron]
                        + [sum(correct_count)]
                    ]

                    score = pd.DataFrame(rows, columns=one_neuron_file.columns)
                    print("scoreee:", rows)

                    result = (
                        pd.concat([one_neuron_file, score])
                        .drop_duplicates(["index"], keep="last")
                        .sort_values("index")
                    )
                    results_df = result.reset_index(drop=True)
                    print("reesuultt:", results_df)

                    results_df.to_excel(PATH + file_name, index=False)

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
