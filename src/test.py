# GENERATE ESTIMATES ON TEST DATA


# predict on test set
prediction = model.predict(X_test)
print("Predictions on test data are done")

# create output file with results
estimation = pd.concat([test_values[building_id], prediction],axis=1)
estimation.to_csv('./reports/EQD_Estimation.csv')