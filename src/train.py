import pandas as pd
import pickle
import time
import os
from sklearn import tree
#from sklearn.ensemble import GradientBoostingClassifier

from helper.conf import *
from functions.helper import change_datatype, load_data
from functions.encoder import encoder
from functions.helper import load_data, change_datatype
from functions.train_model import train_model
from functions.test import test_data



#### track processing time ####
start_time_all = time.time()
print("--- This is the beginning. ---")

#------------------------------#
##   LOAD and CONVERT data    ##
#------------------------------# 

# load data for training
train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)
test_values = load_data(TEST_VALUES_PATH)
print ("--- The data is loaded.")

# use test values for encoding
X_all_raw = pd.concat([train_values, test_values], axis=0, sort=False)
# drop index column
X_all_raw = X_all_raw.reset_index(drop=True)
# convert strings to integer where necessary
X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)


#-------------------------#
##      ENCODE data      ##
#-------------------------#

# Encode data
print("Encoding the data:")
encoded_data = encoder(X_all_shape, train_labels, string_columns, one_hot_columns)

# Save encoded data for post-processing
encoded_data.to_csv(encoded_data_path, index=False)
print ("--- The data is encoded.")


#-------------------------#
##      TRAIN data       ##
#-------------------------#

# Prepare data for training
X_train = encoded_data.loc[:len(train_values)-1, :]
X_test = encoded_data.loc[len(train_values):, :]
y_train = train_labels["damage_grade"]

# Alternative model:
#model = GradientBoostingClassifier(random_state=42)
#param_grid = {
#    "n_estimators": [100],
#    "learning_rate": [1],
#    "min_samples_leaf": [5],
#    }

# Define model
model = tree.DecisionTreeClassifier()
param_grid = {
    "max_depth": [2, 4, 6, 10],
    "min_samples_leaf": [2, 4, 6],
    "criterion" : ['gini'],
    # gini is seemingly faster than entropy at equal performance at score
    }
print("--- The model is set up.")

# Fit the train data to train labels
print("--- Training the model...")
fitted_model = train_model(X_train, y_train, model, param_grid)


#-------------------------#
##  Estimation of data   ##
#-------------------------#

# Save the model to disk
pickle.dump(fitted_model, open(FITTED_MODEL_PATH, 'wb'))
os.rename('./models/model.pickle', './models/model' + str(time_of_run) + '.pickle')

# Estimate labels on test values
test_data(X_test, fitted_model)
print ("--- The estimation is saved to reports.")

#### Track processing time ####
end_time_all = time.time()
print("--- %s seconds for the program to run ---" % (end_time_all - start_time_all))
print("--- This is the end. ---")