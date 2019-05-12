"""HMB While I write algorithms"""

import os
import pickle
import numpy as np
from sklearn.svm import LinearSVC

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler


class ModelSVM:
    def __init__(self, kernel="linear"):
        self.kernel = kernel
        self.params_grid = {
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
        }
        self.model_name = './svm_model_0_0_epi.pkl'

    def fit(self, p_object, verbose=False):
        if os.path.exists(self.model_name):
            with open(self.model_name, 'rb') as f:
                print("[*] Model Already Trained - Loaded...!")
                self.model = pickle.load(f)
            return True

        if self.kernel == "linear":
            model = LinearSVC(class_weight="balanced", C=10.0, max_iter=100000)
        else:
            model = SVC(kernel="rbf", C=5.0, gamma=0.01)

        print("[.] Getting Train data")
        X, Y = p_object.get_train_data_by_episode()
        print("[*] Got Train Data:")
        print("[.] X: {}".format(str(X.shape)))
        print("[.] Y: {}".format(str(Y.shape)))
        
        self.scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
        X = self.scaling.transform(X)
        # X_test = scaling.transform(X_test)

        print("[*] Training the model...")
        model.fit(X, Y.ravel())

        with open(self.model_name, 'wb') as f:
            print("[*] Trained Model Saving | Name: {}".format(self.model_name))
            pickle.dump(model, f)

        self.model = model

    def score(self, p_object):
        print("[*] Calculating Score on Test Episodes: Getting Test Data...")
        
        X, Y = p_object.get_test_data_by_episode()
        print("[*] Got Test Data:")
        print("[.] X: {}".format(str(X.shape)))
        print("[.] Y: {}".format(str(Y.shape)))

        print("[.] Predicting Test Data")
        y_pred = self.model.predict(X)
        fscore = f1_score(y_pred, Y.ravel(), average=None)

        print("[*] F-Score: {}".format(str(fscore)))

        return fscore



    def score2(self, p_object):
        print("evaluating!")

        correct = 0
        total = 0

        _X = p_object.get_test_data_for_comp2()
        print(_X.shape)
        outputs = self.model.predict(_X)
        
        print(outputs.shape)

        no_fold = 30909+1
        print(pred_labels.shape)
        csvData = np.ndarray(shape=(30910, 2))

        for i in range(0, no_fold):
            print(i)
            csvData[i] = np.array([int(i), int(pred_labels[i])]).astype(int)
            # print(time.time())

        os.chdir("..")
        # print(time.time())
        np.savetxt("sub.csv", csvData.astype(int), delimiter=',', fmt="%i %i")
