
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics


def train_model(X, y, model):
    start_time = time.time()
    # fit model
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    # check processing time of model
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"The F1-Score of the training is {scores}")
    return model.fit(X,y)