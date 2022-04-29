from helper.conf import time_of_run

def test_data(X, model):

    """
    This function predicts the labels for the values of the test set. 
    The fittet model is applied and the results saved in records. 
    """

    # predict on test set
    prediction = model.predict(X)
    print("Predictions on test data are done")

    # create output files with results (official result & model report)
    estimation = X[["building_id"]].copy()
    estimation["damage_grade"] = prediction
    estimation.to_csv('./reports/EQD_Estimation' + str(time_of_run) + '.csv', index=False)
    with open('./models/EQD_Estimation' + str(time_of_run) + '.txt', 'w') as f:
        f.write('EQD_Estimation' + str(time_of_run))
    pass
    