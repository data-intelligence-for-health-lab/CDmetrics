# CD_metrics
Case Difficulty (Instance Hardness) metrics in Python, with three ways to measure the difficulty of individual cases: CDmc, CDdm, and CDpu.

## Case Difficulty Metrics
- Case Difficulty Model Complexity **(CDmc)**
  - CDmc is based on the complexity of the neural network required for accurate predictions.

- Case Difficulty Double Model **(CDdm)**
  - CDdm utilizes a pair of neural networks: one predicts a given case, and the other assesses the likelihood that the prediction made by the first model is correct.

- Case Difficulty Predictive Uncertainty **(CDpu)**
  - CDpu evaluates the variability of the neural network's predictions.


## Getting Started
CD_metrics employs neural networks to measure the difficulty of individual cases in a dataset. The metrics are tailored to different definitions of prediction difficulty and are designed to perform well across various datasets.


### Installation
The package was developed using Python. Below, we provide standard installation instructions and guidelines for using CD_metrics to calculate case difficulty on your own datasets.

_For users_
```
pip install CD_metrics
```

#### Anaconda environment

We **strongly recommend** using a separate Python environment. We provide an env file [environment.yml](./environment.yml) to create a conda env with all required dependencies:

```
conda env create --file environment.yml
```

### Usage

Each metric requires certain parameters to run.

- CDmc requires number_of_NNs (the number of neural network models to predict the test case):
```
from difficulty.metrics_utils import model_complexity_score
CDmc_main(data, number_of_NNs, target_column, params)
```

- CDdm requires num_folds (the number of folds to divide the data):
```
from difficulty.metrics_utils import  double_model_score
CDdm_main(data, num_folds, target_column)
```

- CDpu requires number_of_predictions (the number of prediction probabilities to generate):
```
from difficulty.metrics_utils import prediction_uncertainty_score
CDpu_main(data, target_column, number_of_predictions)
```

The hyperparameters are tuned using HyperOptSearch with Ray.
To modify the hyperparameter methods and search space, update the values in metrics_utils/utils.py.

### Guidelines for input dataset

Please follow the recommendations below:

* Only `xlsx` files are accepted.

* The dataset should be preprocessed (scaling, imputation, and encoding must be done before running CD_metrics).

* Do not include any index column.

* The target column name must be clearly specified..

* The metrics only support classification problems based on tabular data.

## Citation

If you're using CD_metrics in your research or application, please cite our [paper](https://www.nature.com/articles/s41598-024-61284-z):

> Kwon, H., Greenberg, M., Josephson, C.B. and Lee, J., 2024. Measuring the prediction difficulty of individual cases in a dataset using machine learning. Scientific Reports, 14(1), p.10474.

```
@article{kwon2024measuring,
  title={Measuring the prediction difficulty of individual cases in a dataset using machine learning},
  author={Kwon, Hyunjin and Greenberg, Matthew and Josephson, Colin Bruce and Lee, Joon},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={10474},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
