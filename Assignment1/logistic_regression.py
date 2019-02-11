import math
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from scipy.special import expit


class Datafiles:
    def __init__(self, fileX, fileY):
        self.fileX = fileX
        self.fileY = fileY
        self.process()

    def process(self):
        # Read from CSV file
        csvfileX=open(self.fileX, 'r')
        csvfileY=open(self.fileY, 'r')

        X = list(csv.reader(csvfileX))
        Y = list(csv.reader(csvfileY))

        # Convert list format to float format
        samples = len(X)
        X1 = []
        Y1 = []
        for i in range(samples):
            X1.append([float(X[i][0]), float(X[i][1])])
            Y1.append(float(Y[i][0]))

        X = X1
        Ylist = Y1

        # Normalize the data (X in R^2)
        meanX = [0, 0]
        varianceX = [0, 0]

        for i in range(samples):
            for j in range(len(meanX)):
                meanX[j] += (X[i][j])
        for j in range(len(meanX)):
            meanX[j] /= samples

        for i in range(samples):
            for j in range(len(meanX)):
                varianceX[j] += (X[i][j] - meanX[j])**2
        for j in range(len(meanX)):
            varianceX[j] /= samples
            varianceX[j] = math.sqrt(varianceX[j])

        for i in range(samples):
            for j in range(len(meanX)):
                X[i][j] = (X[i][j] - meanX[j])/varianceX[j]

        print(meanX)
        print(varianceX)

        partition = {1: {'x': [], 'y': []}, 0: {'x':[], 'y':[]}}
        for i in range(samples):
            if Ylist[i]:
                partition[0]['x'].append(float(X[i][0]))
                partition[0]['y'].append(float(X[i][1]))
            else:
                partition[1]['x'].append(float(X[i][0]))
                partition[1]['y'].append(float(X[i][1]))


        X = np.array(X)
        # Insert the X0 term (all ones: constant)
        X = np.insert(X, 0, 1.0, axis=1)
        Y = np.array(Ylist)

        self.X = X
        self.Y = Y 
        self.partition = partition
        self.samples = samples
        self.features = len(meanX)

data = Datafiles("./data/logisticX.csv", "./data/logisticY.csv")

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


threshold = 0.00001
_iter = 0
theta = np.array([0, 0, 0])

while True:
    htheta = np.matmul(data.X, theta)
    gtheta = sigmoid(htheta)
    gradient = np.dot(data.X.T, (data.Y - gtheta))

    # Calculate Hessian
    H = np.zeros((data.features+1, data.features+1))
    for i in range(data.samples):
        H = H - (sigmoid_derivative(htheta[i]))*(np.outer(data.X[i].T, data.X[i]))
    _iter += 1

    theta_old = theta
    # Update theta

    theta = theta - np.matmul(np.linalg.inv(H), gradient)

    # Check if converged
    if(np.linalg.norm(theta - theta_old) < threshold):
        break;

print("[*] Number of iterations {i}".format(i=_iter))
print(theta)


def plot_theta(data, theta):
    fig = plt.figure(figsize=(12, 10))
    foot = fig.add_subplot(1,1,1)

    foot.scatter(data.partition[0]['x'], data.partition[0]['y'], label="y=0 | Training Data", color= "red", marker= ".")
    foot.scatter(data.partition[1]['x'], data.partition[1]['y'], label="y=1 | Training Data", color= "green", marker= "+")

    x=np.linspace(min(min(data.partition[0]['x']), min(data.partition[1]['x'])), max(max(data.partition[0]['x']), max(data.partition[0]['x'])), data.samples)
    y = -1 * (theta[1]*x + theta[0])/theta[2]

    foot.plot(x, y, label="Decision boundary", color ='blue')

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    # plt.ylim(bottom=0)

    plt.legend()
    plt.title('Q3(a) - Logistic Regression: Using Newton\'s Method')

    plt.savefig('./graphs/logistic_regression_newton_method.png')


plot_theta(data, theta)


