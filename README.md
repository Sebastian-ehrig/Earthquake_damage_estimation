Earthquake_predictor
==============================

Predicting the level of damage to buildings caused by the Gorkha earthquake depending on building location and building construction. The Data that is used for the model predictions are based on surveys conducted by Kathmandu Living Labs and the Central Bureau of Statistics.

This model was developed during the DSR (DataScienceRetreat) Batch30.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Models generated.  
    │   └── Model adjustments    <- Contains log-files for keeping track of model refinements.
    │
    ├── reports            <- Generated analysis as csv.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── functions      <- Scripts to turn raw data into features for modeling
    │   │   │                 
    │   │   ├── encoder.py        <- Function to encode the raw data.
    │   │   ├── helper.py         <- Functions for loading and conversion of the data.
    │   │   ├── test.py           <- Contains function for testing the model.
    │   │   └── train_model.py    <- Function for training the model.
    │   │   
    │   ├── helper             <- Scripts to train models and then use trained models to make 
    │   │   │                     predictions
    │   │   └── conf.py        <- Contains all file-paths and data-keys
    │   │
    │   ├── main.py            <- Script to execute the entire analysis and model predicion
    │   └── train.py           <- Script for encoding the data, taining the model and making 
    │                             the final predictions
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

data notations used in the scope of this project:

raw data: 
 train_values 
 train_labels 
 test_values
 X_all_raw (train & test values)

preprocessed data:
 X_train
 y_train
 X_test

modeled data:
 fitted_model (cv on train set)

output data:
 prediction (on test set)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
