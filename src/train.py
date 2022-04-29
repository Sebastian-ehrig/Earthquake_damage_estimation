# TRAIN MODEL AND CROSS-VALIDATE

from functions.train_model import train_model
from sklearn import tree
import catboost as cb

# define model
model = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='entropy')
# model = cb.CatBoostRegressor(learning_rate=0.1,n_estimators=100,max_depth=5)

# fit the train data to train labels
fitted_model = train_model(X_train, y_train, model)
print("Model is trained")

from helper.conf import *
from functions.helper import change_datatype
from functions.helper import load_data
import pandas as pd

# LOAD DATA
train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)
test_values = load_data(TEST_VALUES_PATH)

## combine values of train and test set 
X_all_raw = pd.concat([train_values, test_values], axis=0, sort=False)

# X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)
X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)
print(f"The score on the training data is {f1score}")

# # fitting the data
# fitted_model = train_model(train_values, train_labels, model)
# print("Model is trained")

# # calculate f1-score on training data (not test score, incl. validation & training data)
# f1score = model.predict(X_train)

# print(f"The score on the training data is {f1score}")

# # predict on test set
# prediction = model.predict(X_test)
# print("Predictions on test data are done")
