# CDmetrics
Case Difficulty (Instance Hardness) metrics in Python, with three ways to measure the difficulty of individual cases: CDmc, CDdm, and CDpu.

## Case Difficulty Metrics
- Case Difficulty Model Complexity **(CDmc)**
  - CDmc is based on the complexity of the neural network required for accurate predictions.

- Case Difficulty Double Model **(CDdm)**
  - CDdm utilizes a pair of neural networks: one predicts a given case, and the other assesses the likelihood that the prediction made by the first model is correct.

- Case Difficulty Predictive Uncertainty **(CDpu)**
  - CDpu evaluates the variability of the neural network's predictions.


## Getting Started
CDmetrics employs neural networks to measure the difficulty of individual cases in a dataset. The metrics are tailored to different definitions of prediction difficulty and are designed to perform well across various datasets.


### Installation
The package was developed using Python. Below, we provide standard installation instructions and guidelines for using CDmetrics to calculate case difficulty on your own datasets.

_For users_
```
pip install cdmetrics
```

_For developers_
```
git clone https://github.com/data-intelligence-for-health-lab/CDmetrics.git
```

#### Anaconda environment

We **strongly recommend** using a separate Python environment. We provide an env file [environment.yml](./environment.yml) to create a conda environment with all required dependencies:

```
conda env create --file environment.yml
```

### Usage

Each metric in CDmetrics requires specific parameters to run.

- CDmc
  - custom: A custom hyperparameter space, and specific training parameters
  - number_of_NNs: The number of neural network models used to make predictions.
  - resources: The number of CPUs for multi-processing.
```
from CDmetrics import CDmc
custom = {}
number_of_NNs = 20
resources = {"CPU": 10, "GPU": 0}

CDmc.compute_metric(data, number_of_NNs, target_column_name, custom, resources)
```

- CDdm
  - custom: A custom hyperparameter space, and specific training parameters
  - num_folds: The number of folds to divide the data.
  - resources: The number of CPUs and GPUs for hyperparameter tuning with Ray.
```
from CDmetrics import CDdm
custom = {}
number_of_folds = 5
resources = {"CPU": 10, "GPU": 0}

CDdm.compute_metric(data, num_folds, target_column_name, custom, resources)
```

- CDpu
  - custom: A custom hyperparameter space, and specific training parameters
  - number_of_predictions: The number of prediction probabilities to generate.
  - resources: The number of CPUs for multi-processing, and CPUs and GPUs for hyperparameter tuning with Ray.
```
from CDmetrics import CDpu
custom = {}
number_of_predictions = 100
resources = {"CPU": 10, "GPU": 0}

CDpu.compute_metric(data, target_column_name, number_of_predictions, custom, resources)
```

#### Hyperparameter Tuning and Customization
For CDdm and CDpu, grid search is supported for hyperparameter tuning.
Users can modify the search space and model configuration by changing the parameter custom.
The parameters that users can customize, along with their default values, are listed below:
  - max_layers(Maximum number of hidden layers):3
  - max_units(Maximum number of units per hidden layer):3
  - learnRate(Learning rate):[0.01, 0.03, 0.1]
  - batch_size(Batch size):[32, 64, 128]
  - activation(Activation function):[relu, tanh]
  - metric(Metric to evaluate the model during training and testing):accuracy
  - validation_split(Train-validation split ratio):0.3
  - epochs(Epochs):100


For CDmc, users can customize the search space and model configuration by changing the parameter custom.
The parameters that users can customize, along with their default values, are listed below:
  - learnRate(Learning rate):0.01
  - batch_size(Batch size):64
  - activation(Activation function):relu
  - metric(Metric to evaluate the model during training and testing):accuracy
  - validation_split(Train-validation split ratio):0.3
  - epochs(Epochs):100



### Guidelines for input dataset

Please follow the recommendations below:

* The dataset should be preprocessed (scaling, imputation, and encoding must be done before running CDmetrics).
* Data needs to be passed in a dataframe.
* Do not include any index column.
* The target column name must be clearly specified.
* The metrics only support classification problems with tabular data.
* CDmc requires data with more than 100 cases to run.

## Citation

If you're using CDmetrics in your research or application, please cite our [paper](https://www.nature.com/articles/s41598-024-61284-z):

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
