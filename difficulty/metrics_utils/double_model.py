import pandas as pd
from difficulty.metrics_utils.utils import tune_parameters
import numpy as np

from sklearn.model_selection import KFold
from  difficulty.metrics_utils.nn import NN

def compute_metric(data, num_folds,target_column):
     
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_order = [[(i + j) % num_folds for i in range(num_folds)] for j in range(num_folds)]
    fold_index = [test_index for _, test_index in kfold.split(data)]
    difficulity = []
    difficulity_index = []

    for folds in fold_order:
        
        train_data_index = []
        for index in  folds[0:len(fold_order)//2]:
            train_data_index += list(fold_index[index])

        evaluation_data_index = []
        for index in  folds[len(fold_order)//2:len(fold_order)-1]:
            evaluation_data_index += list(fold_index[index])

        test_data_index = fold_index[folds[len(fold_order)-1]]

        train_data = data.iloc[train_data_index]
        evaluation_data = data.iloc[evaluation_data_index]
        test_data = data.iloc[test_data_index]
        best_model_A = NN(tune_parameters(NN.tune,train_data,target_column)).model
        evaluation_data_x = evaluation_data.drop(columns=[target_column], axis=1)
        evaluation_data_y = evaluation_data[target_column]
        best_model_A_predictions = best_model_A.predict(evaluation_data_x)
        correctness_df = pd.DataFrame((np.array(best_model_A_predictions) != np.array(evaluation_data_y))*1,index=evaluation_data_x.index)
        best_model_B = NN(tune_parameters(NN.tune,evaluation_data_x,correctness_df))
        test_data_x = test_data.drop(columns=[target_column], axis=1)
        predicted_difficulty = 1 - best_model_B.predict(test_data_x)
        difficulity += predicted_difficulty
        difficulity_index =+ test_data_index
        
    return   pd.DataFrame(difficulity, index=difficulity_index)

