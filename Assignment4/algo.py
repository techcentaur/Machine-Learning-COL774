"""HMB While I write algorithms"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class ModelSVM:
    def __init__(self, kernel="linear"):
        self.kernel = kernel
        self.params_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        }

    def fit(X, Y, verbose=False):
        model = GridSearchCV(SVC(kernel=self.kernel, class_weighted=''),
                             params_grid,
                             cv=10)
        model = model.fit(X, Y)
        if verbose:
            print("[*] Best Estimators from Grid Search: {}".format(
                model.best_estimator_))

        self.model = model

    def score(X, Y):
        accuracy = self.model.score(X, Y)
        return accuracy

    def predict(X):
        return self.model.predict(X)
