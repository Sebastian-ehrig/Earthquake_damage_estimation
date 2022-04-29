import pandas as pd
import pickle
import time
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

from helper.conf import *
from functions.helper import change_datatype, load_data
from functions.encoder import encoder
from functions.helper import load_data, change_datatype
from functions.train_model import train_model



#### track processing time ####
start_time_all = time.time()


#------------------------------#
##   LOAD and CONVERT data    ##
#------------------------------# 

# load data for training
train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)
test_values = load_data(TEST_VALUES_PATH)
print ("The data is loaded")

# use test values for encoding
X_all_raw = pd.concat([train_values, test_values], axis=0, sort=False)
# drop index column
X_all_raw = X_all_raw.reset_index(drop=True)
# convert strings to integer where necessary
X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)
print ("change datatype")

#-------------------------#
##      ENCODE data      ##
#-------------------------#
encoded_data = encoder(X_all_shape, train_labels, string_columns)

# save encoded data for post-processing
encoded_data.to_csv(encoded_data_path, index=False)
print ("The data is encoded")

#-------------------------#
##      TRAIN data       ##
#-------------------------#

# Prepare data for training
X_train = encoded_data.loc[:len(train_values)-1, :]
X_test = encoded_data.loc[len(train_values):, :]
y_train = train_labels["damage_grade"]

# select model

# model = GradientBoostingClassifier(random_state=42)
#param_grid = {
#    "n_estimators": [10, 50, 100],
#    "learning_rate": [0.01, 0.1, 1],
#    "max_depth": [1, 2, 3],
#    }

model = tree.DecisionTreeClassifier(random_state=42)
param_grid = {
    "max_depth": [1, 2, 3, 7],
    "min_samples_leaf": [1, 3, 5],
    }
print("The model is set up")

# fit the train data to train labels
fitted_model = train_model(X_train, y_train, model, param_grid)


#-------------------------#
##  Estimation of data   ##
#-------------------------#

# save the model to disk
pickle.dump(fitted_model, open(FITTED_MODEL_PATH, 'wb'))

print ("The estimation is saved to reports")

#### track processing time ####
end_time_all = time.time()
print("--- %s seconds for the program to run ---" % (end_time_all - start_time_all))