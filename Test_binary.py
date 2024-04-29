import pandas as pd
from sklearn.preprocessing import LabelEncoder
from d_case_difficulty_metrics.metrics import CDdm, CDpu
from d_case_difficulty_metrics.compute_case_difficulty.processing import preprocessor

data = pd.read_csv("./d_case_difficulty_metrics/data/Telco.csv")
# Drop Customer ID
df = data.drop(columns=["customerID"], axis=1)
df["Churn"] = LabelEncoder().fit_transform(df["Churn"])


df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
df.loc[df["tenure"] == 0, "TotalCharges"] = 0
df["TotalCharges"] = df["TotalCharges"].astype("float64")

# Categorical columns and numeric columns
cat_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
num_columns = ["tenure", "TotalCharges", "MonthlyCharges"]
target_column = "Churn"

"""
max_eval_a = 10
max_eval_b = 5
CDdm.CDdm_run(df, max_eval_a, max_eval_b, cat_columns, num_columns, target_column)
"""


hyper_file_name = "CDpu_hyperparam_binary.xlsx"
processing = preprocessor(cat_columns, num_columns)
CDpu.hyperparm_searching(hyper_file_name, df, processing, target_column)


"""
file_name = 'Telco_100.xlsx'
hyper_file_name = 'telco_hyperparam.xlsx'
number_of_predictions = 3
number_of_cpu = 2
processing = preprocessor(cat_columns, num_columns)


if __name__ == '__main__':
    CDpu.multipool_predictions(file_name, hyper_file_name, df, processing,
    target_column, number_of_predictions, number_of_cpu)
"""
