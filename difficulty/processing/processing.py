from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


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

    processor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ]
    )
    return processor



def get_data_col_types(data_frame:pd.DataFrame):
    """
    
    """
    numeric_columns= data_frame._get_numeric_data().columns
    categorical_colunms = list(set(data_frame.columns)- set(numeric_columns))

    return  numeric_columns, categorical_colunms