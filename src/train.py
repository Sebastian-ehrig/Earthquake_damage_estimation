
from functions.train_model import train_model

# Model specification
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='entropy')

# Fitting the data
fitted_model = train_model(train_values, train_labels, model)
