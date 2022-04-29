import pandas as pd
from sklearn import tree
from operator import index
from helper.conf import * 
from functions.helper import change_datatype, load_data
from functions.encoder import encoder
from functions.helper import load_data, change_datatype
from functions.train_model import *


#------------------------------#
##   LOAD and CONVERT data    ##
#------------------------------# 

# load data for training
train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)
test_values = load_data(TEST_VALUES_PATH)
print ("loaded data")

# use test values for encoding
X_all_raw = pd.concat([train_values, test_values], axis=0, sort=False)
# drop index column
X_all_raw.reset_index(drop=True)
# convert strings to integer where necessary
X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)

#-------------------------#
##      ENCODE data      ##
#-------------------------#
encoded_data = encoder(X_all_shape, train_labels, string_columns)
a=1
# split encoded data 
encoded_string_columns = encoded_data[string_columns]
encoded_int_columns = encoded_data[int_columns]

# save encoded data for post-processing
encoded_string_columns.to_csv(encoded_string_col_path, index=False)
encoded_int_columns.to_csv(encoded_int_col_path, index=False)
encoded_data.to_csv(encoded_data_path, index=False)

#-------------------------#
##      TRAIN data       ##
#-------------------------#

y_train = train_labels["damage_grade"]

# select model
model = tree.DecisionTreeClassifier(random_state=42)
# model = cb.CatBoostRegressor(learning_rate=0.1,n_estimators=100,max_depth=5)
# from xgboost import XGBClassifier
# model = XGBClassifier(booster='gbtree')
param_grid = {
    "max_depth": [1, 2, 3, 7],
    "criterion": ['gini', 'entropy'],
    "min_samples_leaf": [1, 3, 5],
    }
print ("set-up model")

# fit the train data to train labels
fitted_model = train_model(X_train, (y_train-1), model, param_grid)

# save the model to disk
import pickle
pickle.dump(fitted_model, open(FITTED_MODEL_PATH, 'wb'))
