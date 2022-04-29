
X_all_raw

import category_encoders as ce
import pandas as pd
import numpy as np 

# def encoder(data)
# load data
TRAIN_VALUES_PATH = "./data/raw/train_values.csv"
TRAIN_LABELS_PATH = "./data/raw/train_labels.csv"
TEST_VALUES_PATH = "./data/raw/test_values.csv"

train_values = pd.read_csv(TRAIN_VALUES_PATH,index_col=0)
train_labels = pd.read_csv(TRAIN_LABELS_PATH,index_col=0)
test_values = pd.read_csv(TEST_VALUES_PATH,index_col=0)

## make report on values
# from pandas_profiling import ProfileReport
# prof = ProfileReport(X_all_raw)
# prof.to_file(output_file='./reports/data_report.html')

data = pd.concat([train_values, test_values], axis=0, sort=False)
data_head = data.head(10)
data_tail = data.tail(10)
data_shape = data.shape
data_dtypes = data.dtypes
data_LSC = data.land_surface_condition.value_counts()
data_ft = data.foundation_type.value_counts()
data_florr_count = data.count_floors_pre_eq
data_age = data.age.value_counts()

#------------------------------------------------------
#Target encoding
#------------------------------------------------------
ce_te = ce.TargetEncoder(cols=['count_floors_pre_eq'])

#column to perform encoding
X = data.count_floors_pre_eq.astype('int')

#X['count_floors_pre_eq'] = X['count_floors_pre_eq'].astype('int')

y = data.age.astype('int')
#create an object of the Targetencoder
ce_te.fit(X,y)

Target_encoder = ce_te.transform(X).head(10)

#-------------------------------------------------------
# create an object of the BaseNEncoder
#-------------------------------------------------------
ce_baseN4 = ce.BaseNEncoder(cols=['age'],base=2)
# fit and transform and you will get the encoded data
ce_baseN4.fit_transform(data_age).head(10)

pause