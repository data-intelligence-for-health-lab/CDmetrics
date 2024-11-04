from sklearn.datasets import load_breast_cancer, make_classification
import pandas as pd
from CDmetrics import CDmc, CDdm, CDpu
from CDmetrics import utils
import os


bc_df = load_breast_cancer()
df = pd.DataFrame(bc_df["data"], columns=bc_df["feature_names"])
df["target"] = bc_df["target"]


X, y = make_classification(
    n_samples=3000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=3,
    random_state=42,
)
df = pd.DataFrame(X, columns=["x_1", "x_2", "x_3"])
df["target"] = y

df = df[:30]

# number_of_NNs = 3
# metrics_CDmc = CDmc.compute_metric(df, number_of_NNs, "target")
# print(f"Number of NNs: {number_of_NNs}")
# print("metrics_CDmc:", metrics_CDmc)



# number_of_folds = 5
# metrics_CDdm = CDdm.compute_metric(
#     df,
#     number_of_folds,
#     "target",
#     max_layers=1,
#     max_units=2,
#     resources={"CPU": 8, "GPU": 0},
# )
# print("metrics_CDdm:", metrics_CDdm)


metrics_CDpu = CDpu.compute_metric(df, "target", 5, max_layers=1, max_units=2, resources={"CPU": 8, "GPU": 0})
print("metrics_CDpu:", metrics_CDpu)

