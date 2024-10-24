from sklearn.datasets import load_breast_cancer
import pandas as pd
from Case_Difficulty.CDmetrics import CDmc, CDdm, CDpu
from Case_Difficulty.CDmetrics import utils


data_bc = load_breast_cancer()

bc_df = pd.DataFrame(data_bc["data"], columns=data_bc["feature_names"])
bc_df["target"] = data_bc["target"]
available_resources = utils.get_available_resources()

metrics_CDdm = CDdm.compute_metric(
    bc_df, 11, "target", max_layers=2, max_units=10, resources={"CPU": 8, "GPU": 0}
)

metrics_CDpu = CDpu.compute_metric(
    bc_df, "target", 100, max_layers=2, max_units=10, resources={"CPU": 8, "GPU": 0}
)

metrics_CDmc = CDmc.compute_metric(bc_df, 5, "target")
