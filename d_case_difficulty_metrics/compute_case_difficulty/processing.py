from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import pandas as pd
import sys


def preprocessor(categorical_columns, numeric_columns):
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ]
    )
    return preprocessor


def check_curr_status(n_samples, file_name):
    if not os.path.isfile(file_name):
        print("File does not exist, starting from index 0.")
        return 0
    try:
        data = pd.read_excel(file_name)

    except Exception:
        print("Error reading the file")
        sys.exit(0)

    starting = len(data)
    if starting == 0:
        print("Starting index: 0")
        return 0

    if starting == n_samples:
        print("All samples are done.")
        return None

    print("Starting index:", starting)
    return starting
