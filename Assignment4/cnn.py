import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

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

    def fit(self, p_object):
        """p_object: Is the object of Processing class"""

        self.model = ConvNet(self.num_classes).to(self.device)
        print("\n[#] Model defined [#]\n")
        print(self.model)

        # Loss and optimizer
        criterion = nn.NLLLoss() # negative log liklihood
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            # get data by generator
            episode = 0
            for X, Y in p_object.get_train_data_with_rgbchannel():
                max_batches = len(X)//self.batch_size
                # print(max_batches)
                batch = 0
                for i in range(max_batches):
                    batch += 1
                    _X = X[i*self.batch_size: (i+1)*self.batch_size]
                    _Y = Y[i*self.batch_size: (i+1)*self.batch_size]

                    _Y = [[1, 0] if _Y.ravel()[i] == 0 else [0, 1] for i in range(_Y.shape[0])]
                    _Y = np.array(_Y)

                    _X = Variable(torch.from_numpy(_X)).float()
                    _Y = Variable(torch.from_numpy(_Y).long())
                    _Y = torch.max(_Y, 1)[1]
                    
                    # Forward pass
                    outputs = self.model(_X)
                    loss = criterion(outputs, _Y)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # track the f-score
                    # print(_Y)
                    # print(torch.max(outputs, 1)[1])
                    score = (f1_score(_Y, torch.max(outputs, 1)[1], average=None))
                    print("[.] Batch {b} | F1-Score: {s}".format(b=batch, s=str(score)))
                    # wait()

                episode += 1
                if episode > 3:
                    break
                print ('Epoch [{}/{}], Episode [{}/{}], Loss: {:.4f}' 
                       .format(epoch, self.num_epochs, episode, len(p_object.train_episodes), loss.item()))
            # break

    def score(self, p_object):
        self.model.eval()

        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            first=True
            i = 0
            for X, Y in p_object.get_test_data_with_rgbchannel():
                i += 1
                if i>4:
                    break
                if first:
                    _X = X 
                    _Y = Y
                    first=False
                    continue
                _X = np.concatenate((_X, X), axis=0)
                _Y = np.concatenate((_Y, Y), axis=0)

            _X = Variable(torch.from_numpy(_X)).float()
            _Y = Variable(torch.from_numpy(_Y).long())

            outputs = self.model(_X)

            total = _Y.size(0)
            _Y = torch.from_numpy((_Y.numpy()).T[0])
            correct = (torch.max(outputs.data, 1)[1] == _Y).sum().item()

            score = (f1_score(_Y, torch.max(outputs.data, 1)[1], average=None))
            print("[*] F1-Score: {s}".format(s=str(score)))


    def save(self, name=''):
        torch.save(model.state_dict(), 'model_nn_{}.ckpt'.format(name))
