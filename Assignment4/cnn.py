import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import sys

from sklearn.model_selection import GridSearchCV
# from skorch import NeuralNetClassifier

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(250, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))

        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ModelCNN:
    def __init__(self, param):
        print("[*] Running CNN Model: ")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        
        self.num_epochs = param["num_epoches"] #10
        self.num_classes = param["num_classes"]
        self.batch_size = param["batch_size"] #50
        self.learning_rate = param["learning_rate"] #.001

    def fit(self, p_object):
        """p_object: Is the object of Processing class"""

        self.model = ConvNet(self.num_classes).to(self.device)
        print(self.model)

        # Loss and optimizer
        criterion = nn.NLLLoss() # negative log liklihood
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            # get data by generator
            episode = 0
            for X, Y in p_object.get_train_data_by_episode():
                max_batches = len(X)//self.batch_size
                for i in range(max_batches):
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

            
                episode += 1
                print ('Epoch [{}/{}], Episode [{}/{}], Loss: {:.4f}' 
                       .format(epoch, self.num_epochs, episode, len(p_object.train_episodes), loss.item()))

    def score(self, p_object):
        self.model.eval()

        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            first=True
            for X, Y in p_object.get_test_data_by_episode():
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
            correct = (torch.max(outputs, 1)[1] == _Y).sum().item()

            print(total)
            print(correct)
            print('Accuracy on test images: {} %'.format(100 * correct / total))

    def save(self, name=''):
        torch.save(model.state_dict(), 'model_nn_{}.ckpt'.format(name))
