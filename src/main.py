import pandas as pd
import os


###########################################

# LOAD DATA
import load
load


# PREPROCESSING

 ## encode data
import functions.encoder
encoder

 ## split data set


# TRAIN MODEL
 ## specify model
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='entropy')

 ## fit model, validate & predict on test set
import train
train


# CREATE EQD ESTIMATION

estimation = pd.concat([test_values[building_id], prediction],axis=1)
estimation.to_csv('./reports/EQD_Estimation.csv')

############################################