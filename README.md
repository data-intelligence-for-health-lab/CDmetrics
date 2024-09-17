# CD_metrics
Case difficulty (Instance Hardness) metrics in Python, with three ways to measure the difficulty of individual cases: CDmc, Ddm, and CDpu.

## Case Difficulty Metrics
- Case Difficulty Model Complexity **(CDmc)**
  - CDmc is based on the complexity of the neural network required for accurate predictions.

- Case Difficulty Double Model **(CDdm)**
  - CDdm utilizes a pair of neural networks: one predicts a given case, and the other assesses the likelihood that the prediction made by the first model is correct.

- Case Difficulty Predictive Uncertainty **(CDpu)**
  - CDpu evaluates the variability of the neural network's predictions.


## Getting Started
CD_metrics employes neural networks to measure the difficulty of individual cases in a datasets. The metrics were designed to perform well across various datasets and tailored to different definitions of prediction difficulty.


### Installation
The package was developed with the Python. Below, we present the standard installation and guideline to use the CD_metrics to calcaulte case difficulty of own datasets. 

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

Each metrics require specific parameters to run.

- CDmc needs number_of_NNs (Number of neural network models to test)
- CDdm needs num_folds (Number of folds to divide the data)
- CDpu needs number_of_predictions (number of prediction probabilites to generate)



### Guidelines for input dataset

Please follow the recommendations below:

* Only `xlsx` files are accepted

* The dataset should be cleaned with preprocessing (scaling, imputation, and encoding need to be done before run CD_metrics)

* The user need to assign proper categorical columns, numeric columns, and target column
  
* The metrics can only do classification problem


## Citation

If you're using CD_metrics in your research or application, please cite our [paper](https://link):

> Paper info 

```
@article{paiva2022relating,
      title={ },
      author={ },
      journal={ },
      volume={ },
      number={ },
      pages={ },
      year={ },
      publisher={ }
}
```


## References
