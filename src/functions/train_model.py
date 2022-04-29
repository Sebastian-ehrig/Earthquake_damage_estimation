from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import time


def train_model(X, y, model, param_grid):

    gscv = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_micro", refit=True)

    # check processing time of model
    start_time = time.time()
    gscv.fit(X, y)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    # Scores
    score = gscv.best_score_
    print(f"The F1-Score of the training is {score}")

    return gscv.best_estimator_, score