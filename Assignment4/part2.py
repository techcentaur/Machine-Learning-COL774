import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import sys
import os
import csv
import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pickle

from sklearn.preprocessing import MinMaxScaler

# from skorch import NeuralNetClassifier

def wait():
    while True:
        pass

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # feature extraction from here
        self.layer1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))

        # apply fully connected as nn
        self.fc1 = nn.Linear(256, 2048)
        self.do1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 512)
        self.do2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, 256)
        self.do3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(256, 64)
        self.fc5= nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        print(out.shape)
        out = self.fc1(out)
        out = self.do1(out)
        out = self.fc2(out)
        out = self.do2(out)
        out = self.fc3(out)
        out = self.do3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out


class ModelCNN:
    def __init__(self, param):
        print("[*] Running CNN Model: ")

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        
        self.num_epochs = param["num_epoches"] #10
        self.num_classes = param["num_classes"]
        self.batch_size = param["batch_size"] #50
        self.learning_rate = param["learning_rate"] #.001
        self.model_path = "./exp_comp_model.pkl"

    def fit(self, p_object):
        """p_object: Is the object of Processing class"""

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                print("[*] Model Loaded!")
                self.model = pickle.load(f)
            return True

        self.model = ConvNet(self.num_classes).to(self.device)
        print("\n[#] Model defined [#]\n")
        print(self.model)

        weight = [1, 1.5]
        #class_weights = torch.FloatTensor(weight)

        #criterion = nn.NLLLoss(weight=class_weights) # negative log liklihood
        criterion = nn.NLLLoss() # negative log liklihood
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model

        # Train the model
        for epoch in range(self.num_epochs):
            # get data by generator
            episode = 0
            X, Y = p_object.get_train_data_with_rgbchannel_for_comp()
            max_batches = len(X)//self.batch_size

            print("[Scaling data]")
            a, b, c, d = X.shape
            X = X.reshape((a, b*c*d))
            self.scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
            X = self.scaling.transform(X)
            X = X.reshape((a, b, c, d))
            print("[Scaling data]")

            batch = 0
            for i in range(max_batches):
                batch += 1
                _X = X[i*self.batch_size: (i+1)*self.batch_size]
                _Y = Y[i*self.batch_size: (i+1)*self.batch_size]

                _X = Variable(torch.from_numpy(_X)).float()
                _Y = Variable(torch.from_numpy(_Y).long())
                _Y = torch.from_numpy((_Y.numpy()).T[0])
                                
                # Forward pass
                outputs = self.model(_X)
                loss = criterion(outputs, _Y)
                
                # optimizer.zero_grad()
                loss.backward()
                # optimizer.step()
                
                score = (f1_score(_Y, torch.max(outputs, 1)[1], average=None))
                
                print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, F1-Score: {}' 
                       .format(epoch, self.num_epochs, batch, max_batches, loss.item(), str(score)))


        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            print("[*] Model Saved | Name: {}".format(self.model_path))


    def score(self, p_object):
        self.model.eval()

        print("evaluating!")
        with torch.no_grad():
            correct = 0
            total = 0
            _X = p_object.get_test_data_for_comp()
            print("got data")
            _X = Variable(torch.from_numpy(_X)).float()
            outputs = self.model(_X)
            pred_labels = torch.max(outputs.data, 1)[1]
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


    def save(self, name=''):
        torch.save(model.state_dict(), 'model_nn_{}.ckpt'.format(name))
