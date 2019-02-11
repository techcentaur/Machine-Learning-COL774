import numpy as np 
import pandas as pd
from scipy.special import expit
import csv

def sigmoid(z):
    """return the sigmoid value of z(can be a matrix)"""
    return 1/(1 + np.exp(-z))


def get_gradient(X, Y, thetas):
    X_sig = sigmoid(np.matmul(X.T,thetas))
    return np.dot(X.T, (Y - X_sig))

def get_hessian(X, Y, thetas):
    m = X.shape[0]
    n = X.shape[1]-1

    hessian = np.zeros((n+1, n+1))
    htheta = np.matmul(X, thetas.T)
    print(htheta.shape)
    for i in range(m):
        feats = htheta[i]
        print(feats)
        # print(type(np.outer(X[i].T,X[i])))
        hessian = hessian - (np.outer(X[i].T,X[i]))*((sigmoid(feats)*(1-sigmoid(feats))))

    return hessian


def newtons_method(X, Y):
    max_iter = 1
    threshold = 1.0e-4

    m = X.shape[0]  # samples
    n = X.shape[1]-1 # features
    
    thetas = np.zeros([n+1,])

    iteration = 0
    while True:
        gradient = get_gradient(X, Y, thetas)
        print(iteration, " ", thetas)

        #Hessian matrix
        H = get_hessian(X, Y, thetas)

        thetas_old = thetas
        print((np.asmatrix(H)).I.shape)
        print(gradient.shape)
        # update theta as: theta = theta - hessian_inverse * gradient
        thetas = thetas - np.array((np.asmatrix(H)).I @ gradient)
        iteration += 1
        converged = np.linalg.norm(thetas - thetas_old)
        if(converged < threshold or iteration > max_iter):
            break
    

    # print(thetas)
    # print(i)
    print("[*] Logistic Regression by Newton's Method!")




def read_files(filenameX, filenameY):
    csvfile=open("./data/logisticX.csv", 'r')
    x = list(csv.reader(csvfile))
    csvfile=open("./data/logisticY.csv", 'r')
    y = list(csv.reader(csvfile))

    x_=[]
    y_=[]
    for i in range(len(x)):
        x_.append([float(x[i][0]),float(x[i][1])])
        y_.append(float(y[i][0]))
    x = np.array(x_)
    y = np.array(y_)
    x = np.insert(x, 0, 1.0, axis=1)

    return x, y
        
if __name__ == '__main__':
    X, y = read_files("./data/logisticX.csv", "./data/logisticY.csv")
    print(X.shape, " ", y.shape)
    newtons_method(X, y)