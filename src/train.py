from helper.conf import *
from functions.helper import load_data


train_values = load_data(TRAIN_VALUES_PATH)
train_labels = load_data(TRAIN_LABELS_PATH)


from pandas_profiling import ProfileReport
prof = ProfileReport(train_values)
prof.to_file(output_file='./reports/output.html')