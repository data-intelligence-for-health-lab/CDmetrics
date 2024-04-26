import pandas as pd
from sklearn.preprocessing import LabelEncoder
from d_case_difficulty_metrics.metrics import CDdm, CDpu

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

file_name = "./d_case_difficulty_metrics/result/CDpu_hyperparam.xlsx"
CDpu.hyperparm_searching(file_name, df, cat_columns, num_columns, target_column)
