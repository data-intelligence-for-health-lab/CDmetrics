import pandas as pd
from sklearn.preprocessing import LabelEncoder
from d_case_difficulty_metrics.src.metrics import CDmc, CDdm, CDpu
from d_case_difficulty_metrics.src.processing.processing import preprocessor

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


# CDdm
file_name = "Customer_CDdm.xlsx"
max_eval_a = 10
max_eval_b = 5
processing = preprocessor(cat_columns, num_columns)
CDdm.CDdm_run(df, max_eval_a, max_eval_b, processing, target_column)


# CDpu
hyper_file_name = "CDpu_hyperparam_multi.xlsx"
processing = preprocessor(cat_columns, num_columns)
CDpu.hyperparm_searching(hyper_file_name, df, processing, target_column)


file_name = "Customer_CDpu.xlsx"
hyper_file_name = "customer_hyperparam.xlsx"
number_of_predictions = 2
number_of_cpu = 2
if __name__ == "__main__":
    CDpu.CDpu_run(
        file_name,
        hyper_file_name,
        df,
        processing,
        target_column,
        number_of_predictions,
        number_of_cpu,
    )


# CDmc
file_name = "Customer_CDmc.xlsx"
number_of_NNs = 3
processing = preprocessor(cat_columns, num_columns)
if __name__ == "__main__":
    CDmc.CDmc_run(file_name, df, processing, target_column, number_of_NNs)
