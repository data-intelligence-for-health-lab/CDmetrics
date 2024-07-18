import numpy as np

def double_model_score(file_name, data, max_eval_a, max_eval_b,target_column) -> np.Array:
    """
    Compute the difficulity score using the double models approach
    :params 

    :return array  of difficulity scores,  with elements  the difficulty of samples
    
    """
    fold_order = [[(i + j) % 5 for i in range(5)] for j in range(5)]
    Fold_data_save = []
    Fold_difficulty_save = []

    for i in range(len(fold_order)):
        folds = fold_order[i]
        difficulty_data_for_model_B, difficulty = CDdm_main(
            data, folds, max_eval_a, max_eval_b, processing, target_column
        )
        Fold_data_save.append(difficulty_data_for_model_B)
        Fold_difficulty_save.append(difficulty)

    dataframe_temp = Fold_data_save

    # Combine the results
    for i in range(len(fold_order)):
        dataframe_temp[i]["difficulty"] = Fold_difficulty_save[i]

    new_column_names = data.columns.tolist()
    new_column_names.extend(["difficulty"])
    temp_df = pd.DataFrame(
        np.row_stack(
            [
                dataframe_temp[0],
                dataframe_temp[1],
                dataframe_temp[2],
                dataframe_temp[3],
                dataframe_temp[4],
            ]
        ),
        columns=new_column_names,
    )
    temp_df.to_excel(PATH + file_name, header=True, index=False)
    return temp_df



def model_complexity_score(file_name, data, processing, target_column, number_of_NNs) ->np.Array:
    """ 
    compute the difficulty score using the model  complexity approach

    :return array  of difficulity scores,  with elements  the difficulty of samples
    """
    output = multiprocessing.Queue()
    n_samples = len(data)
    print("PATH + file_name:", PATH + file_name)
    starting, curr_df = curr_status(n_samples, PATH + file_name)

    rows_value = []
    rows = []

    # CDmc_set
    try:
        for index in range(starting, n_samples):
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
        if len(curr_df) == n_samples:
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
        MNN = round(len(data) * 0.01)
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

            for repeat_index in repeat_index_neuron_count_view["index"].values.tolist():
                # the test case that want to check the difficulty
                X_test = X_all.iloc[[repeat_index]]
                y_test = y_all.iloc[repeat_index]

                # X,y the dataset wilthout the test case
                X = X_all.drop(index=[repeat_index])
                y = y_all.drop(index=[repeat_index])

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
                    return results_df

                except KeyboardInterrupt:
                    for process in processes:
                        process.terminate()
                        process.join()
                        print("Keyboard error occurred")
                    results_df.to_excel(PATH + "interrupted_" + file_name, index=False)
                    sys.exit()

                except Exception as e:
                    print(f"Error: {e}")
                    print("An error occurred")
                    results_df.to_excel(PATH + "error_" + file_name, index=False)
                    sys.exit(0)


def prediction_uncertainty_score(
    file_name,
    hyper_file_name,
    data,
    processing,
    target_column,
    number_of_predictions,
    number_of_cpu,
):
    

     """ 
    compute the difficulty score using the model  complexity approach

    :return array  of difficulity scores,  with elements  the difficulty of samples
    """
    
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
