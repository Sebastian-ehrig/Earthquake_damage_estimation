
from operator import index
from helper.conf import *
from functions.helper import change_datatype, load_data
from functions.encoder import encoder
import pandas as pd

#------------------------------#
##   LOAD and CONVERT data    ##
#------------------------------# 

train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)
test_values = load_data(TEST_VALUES_PATH)

## combine values of train and test set 
X_all_raw = pd.concat([train_values, test_values], axis=0, sort=False)

# drop index column
X_all_raw.reset_index(drop=True)

# convert strings to integer where necessary
X_all_shape = change_datatype(X_all_raw, string_columns, int_columns)

#-------------------------#
##      ENCODE data      ##
#-------------------------#
encoded_data = encoder(X_all_shape, train_labels, string_columns)

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