import pandas as pd
from sklearn.preprocessing import LabelEncoder
from d_case_difficulty_metrics.metrics import CDmc, CDdm, CDpu
from d_case_difficulty_metrics.compute_case_difficulty.processing import preprocessor

data = pd.read_csv("./d_case_difficulty_metrics/data/Customer.csv")

df = data.drop(columns=["ID"], axis=1)
df["Segmentation"] = LabelEncoder().fit_transform(df["Segmentation"])

cat_columns = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1",
]
num_columns = ["Age", "Work_Experience", "Family_Size"]
target_column = "Segmentation"

"""
max_eval_a = 10
max_eval_b = 5

CDdm.CDdm_run(df, max_eval_a, max_eval_b, cat_columns, num_columns, target_column)
"""

"""
hyper_file_name = "CDpu_hyperparam_multi.xlsx"
processing = preprocessor(cat_columns, num_columns)
CDpu.hyperparm_searching(hyper_file_name, df, processing, target_column)


file_name = 'Customer_100.xlsx'
hyper_file_name = 'customer_hyperparam.xlsx'
number_of_predictions = 2
number_of_cpu = 2
if __name__ == '__main__':
    CDpu.multipool_predictions(file_name, hyper_file_name, df, processing,
    target_column, number_of_predictions, number_of_cpu)
"""


file_name = "./Customer.xlsx"
number_of_NNs = 20

processing = preprocessor(cat_columns, num_columns)
if __name__ == "__main__":
    CDmc.multipool_predictions(file_name, df, processing)
