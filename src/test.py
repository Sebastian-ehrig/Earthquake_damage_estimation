import pickle
import pandas as pd
from helper.conf import *
from functions.helper import load_data, change_datatype

# GENERATE ESTIMATES ON TEST DATA


def test_data(X, model):
    ## Load model

    model = pickle.load(open(FITTED_MODEL_PATH, "rb"))

    ## load test data
    test_values = load_data(TEST_VALUES_PATH)
    values_int = change_datatype(test_values, string_columns, int_columns)
    X_test = values_int.drop(string_columns, axis=1)

    # predict on test set
    prediction = model.predict(X_test)
    print("Predictions on test data are done")

    # create output file with results
    estimation = test_values[["building_id"]].copy()
    estimation["damage_grade"] = prediction
    estimation.to_csv('./reports/EQD_Estimation.csv', index=False)
    