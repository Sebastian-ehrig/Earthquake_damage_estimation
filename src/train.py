# TRAIN MODEL AND CROSS-VALIDATE


from functions.train_model import train_model
from sklearn import tree
import catboost as cb

# define model
model = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='entropy')
# model = cb.CatBoostRegressor(learning_rate=0.1,n_estimators=100,max_depth=5)

# fit the train data to train labels
fitted_model = train_model(X_train, y_train, model)
print("Model is trained")

# calculate f1-score on training data (not test score, incl. validation & training data)
f1score = model.predict(X_train)

print(f"The score on the training data is {f1score}")