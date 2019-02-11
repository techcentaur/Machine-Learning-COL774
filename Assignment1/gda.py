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

        # Alaska - 1 || Canada - 0
        # Convert list format to float format
        samples = len(X)
        X1 = []
        Y1 = []
        for i in range(samples):
            X[i] = X[i][0].split("  ")
            X1.append([float(X[i][0]), float(X[i][1])])
            
            if Y[i][0].lower() == "alaska":
                Y1.append(0)
            else:
                Y1.append(1)

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

        # Data Normalized

        # Parition Data and Calculate Mean and Variance
        partition = {1: {'x': [], 'y': []}, 0: {'x':[], 'y':[]}}
        mu0 = np.array([0., 0.])
        mu1 = np.array([0., 0.])

        phi = 0
        for i in range(samples):
            if Ylist[i]:
                partition[1]['x'].append(float(X[i][0]))
                partition[1]['y'].append(float(X[i][1]))
                mu1 += X[i]
                phi += 1
            else:
                partition[0]['x'].append(float(X[i][0]))
                partition[0]['y'].append(float(X[i][1]))
                mu0 += X[i]

        phi /= samples
        mu0 = mu0 / len(partition[0]['x'])
        mu1 = mu1 / len(partition[1]['x'])

        Xlist = X
        X = np.array(X)
        Y = np.array(Ylist)

        print("[*] Mean of distribution of X (given y=0): {} and (given y=1): {}".format(mu0, mu1))
        print("[*] Parameter of distribution of y: {}".format(phi))

        sigma = np.zeros((X.shape[1], X.shape[1]))
        for i in range(samples):
            if Y[i]:
                sigma += np.outer(X[i] - mu1, X[i] - mu1)
            else:
                sigma += np.outer(X[i] - mu0, X[i] - mu0)

        sigma /= samples
        sigma_inverse = np.linalg.inv(sigma)

        print("[*] Covariance Matrix: {}".format(sigma))

        self.X = X
        self.Y = Y 
        self.partition = partition
        self.samples = samples
        self.features = len(meanX)

data = Datafiles("./data/q4x.dat", "./data/q4y.dat")