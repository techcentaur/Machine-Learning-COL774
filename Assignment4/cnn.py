import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import sys
import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
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
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=2),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))

        # apply fully connected as nn
        self.fc1 = nn.Linear(6912, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

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

        self.model_name = "./cnn_model.pkl"

    def fit(self, p_object):
        if os.path.exists(self.model_name):
            with open(self.model_name, 'rb') as f:
                print("[*] Model Already Trained - Loaded...!")
                self.model = pickle.load(f)
            return True

        self.model = ConvNet(self.num_classes).to(self.device)
        print("\n[#] Model defined [#]\n")
        print(self.model)

        # criterion = nn.NLLLoss(weight=class_weights) # negative log liklihood
        criterion = nn.NLLLoss() # negative log liklihood
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            # get data by generator
            episode = 0
            X, Y = p_object.get_train_data_with_rgbchannel()
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

        with open(self.model_name, 'wb') as f:
            print("[*] Trained Model Saving | Name: {}".format(self.model_name))
            pickle.dump(self.model, f)



    def score(self, p_object):
        self.model.eval()

        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            X, Y = p_object.get_test_data_with_rgbchannel()

            print("[Scaling data]")
            a, b, c, d = X.shape
            X1 = X.reshape((a, b*c*d))
            # self.scaling = MinMaxScaler(feature_range=(-1,1)).fit(X1)
            X1 = self.scaling.transform(X1)
            X = X1.reshape((a, b, c, d))
            print("[Scaling data]")

            X = Variable(torch.from_numpy(X)).float()
            Y = Variable(torch.from_numpy(Y).long())

            outputs = self.model(X)

            total = Y.size(0)
            Y = torch.from_numpy((Y.numpy()).T[0])
            correct = (torch.max(outputs.data, 1)[1] == Y).sum().item()

            score = (f1_score(Y, torch.max(outputs.data, 1)[1], average=None))
            print("[*] F1-Score: {s}".format(s=str(score)))


    def save(self, name=''):
        torch.save(model.state_dict(), 'model_nn_{}.ckpt'.format(name))
