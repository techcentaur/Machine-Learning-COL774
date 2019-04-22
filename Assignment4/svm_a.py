"""HMB While I write algorithms"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import f1_score

class ModelSVM:
    def __init__(self, kernel="linear"):
        self.kernel = kernel
        self.params_grid = {
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
        }

    def fit(self, p_object, verbose=False):
        # model = GridSearchCV(SVC(kernel=self.kernel, class_weight="balanced", gamma="scale"),
        #                      self.params_grid, n_jobs=-1, cv=5,)
        print("[*] Training")
        model = SVC(kernel=self.kernel, class_weight="balanced", gamma="scale")

        # i = -1
        for X, Y in p_object.get_train_data_by_episode():
            # print(i, end=" ")
            # print(Y.ravel())
            model.fit(X, Y.ravel())
            # i+=1

        # if verbose:
        #     print("[*] Best Estimators from Grid Search: {}".format(
        #         model.best_estimator_))

        self.model = model

    def score(self, p_object):
        print("[*] Score:")
        first=True
        for X, Y in p_object.get_test_data_by_episode():
            if first:
                _X = X 
                _Y = Y
                first=False
                continue
            _X = np.concatenate((_X, X), axis=0)
            _Y = np.concatenate((_Y, Y), axis=0)

        y_pred = self.model.predict(_X)
        fscore = f1_score(y_pred, _Y.ravel())

        # print(y_pred)
        # print(_Y.ravel())
        print(fscore)
        return fscore
