import pandas as pd
from sklearn.preprocessing import LabelEncoder
from d_case_difficulty_metrics.metrics import CDdm, CDpu

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

file_name = "./d_case_difficulty_metrics/result/CDpu_hyperparam..xlsx"
CDpu.hyperparm_searching(file_name, df, cat_columns, num_columns, target_column)
