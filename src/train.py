
from functions.train_model import train_model


# fitting the data
fitted_model = train_model(train_values, train_labels, model)
print("Model is trained")

# predict on test set
prediction = model.predict(y_test)
print("Predictions on test data are done")