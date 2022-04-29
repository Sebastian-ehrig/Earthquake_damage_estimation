
from functions.train_model import train_model


# fitting the data
fitted_model = train_model(train_values, train_labels, model)
print("Model is trained")

# calculate f1-score on training data (not test score, incl. validation & training data)
f1score = model.predict(X_train)

print(f"The score on the training data is {f1score}")

# predict on test set
prediction = model.predict(X_test)
print("Predictions on test data are done")