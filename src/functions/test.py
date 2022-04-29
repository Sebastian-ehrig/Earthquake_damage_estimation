# GENERATE ESTIMATES ON TEST DATA
import time

def test_data(X, model):

    # predict on test set
    prediction = model.predict(X)
    print("Predictions on test data are done")

    # create output file with results
    estimation = X[["building_id"]].copy()
    estimation["damage_grade"] = prediction
    timestamp = time.strftime('%Y_%m_%d_%H')
    estimation.to_csv('./reports/EQD_Estimation_' + str(timestamp) + '.csv', index=False)
    
    