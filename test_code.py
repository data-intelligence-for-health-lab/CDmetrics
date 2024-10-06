from sklearn.datasets import load_breast_cancer
import pandas as pd
from difficulty.metrics import double_model, prediction_uncertainty
from difficulty.metrics import utils






data_bc = load_breast_cancer()

bc_df = pd.DataFrame(data_bc['data'], columns=data_bc['feature_names'])
bc_df["target"] = data_bc["target"]

# metrics = double_model.compute_metric(bc_df,11,"target",2,10,{"CPU":8,'GPU':1})
metrics = prediction_uncertainty.compute_metric(bc_df,"target",100,2,10,{"CPU":8,'GPU':1})

# print(utils.get_available_resources())


