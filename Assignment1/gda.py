from math import (log, sqrt)
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
            varianceX[j] = sqrt(varianceX[j])

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
        mu0 /= len(partition[0]['x'])
        mu1 /= len(partition[1]['x'])

        Xlist = X
        X = np.array(X)
        Y = np.array(Ylist)

        print("\n[*] Mean of distribution of X \n(given y=0): {}\n(given y=1): {}\n".format(mu0, mu1))
        print("[*] Parameter of distribution of y: {}\n".format(phi))

        # If both covariance matrices are assumed to be same
        sigma = np.zeros((X.shape[1], X.shape[1]))
        for i in range(samples):
            if Y[i]:
                sigma += np.outer(X[i] - mu1, X[i] - mu1)
            else:
                sigma += np.outer(X[i] - mu0, X[i] - mu0)

        sigma /= samples

        print("[*] Covariance Matrix:\n {}\n".format(sigma))

        # If both covariance matrices are assumed to be different
        sigma0 = np.zeros((X.shape[1], X.shape[1]))
        sigma1 = np.zeros((X.shape[1], X.shape[1]))
        for i in range(samples):
            if Y[i]:
                sigma1 += np.outer(X[i] - mu1, X[i] - mu1)
            else:
                sigma0 += np.outer(X[i] - mu0, X[i] - mu0)

        sigma0 /= len(partition[0]['x'])
        sigma1 /= len(partition[1]['x'])


        print("[*] Covariance Matrix for y=0:\n {}\n".format(sigma0))
        print("[*] Covariance Matrix for y=1:\n {}\n".format(sigma1))


        self.X = X
        self.Y = Y 
        self.partition = partition
        self.samples = samples
        self.features = len(meanX)
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.phi = phi
        self.sigma0 = sigma0
        self.sigma1 = sigma1

class Figures:
    def __init__(self, data, which, plot_on_same=False):
        # Plot GDA linear line separator
       
        self.plot_on_same = False
        if not which:
            self.plot_GDA_with_same_sigma(data)
        else:
            self.plot_GDA_with_different_sigma(data)

        self.plot_on_same = plot_on_same
        if self.plot_on_same:
            if not which:
                self.plot_GDA_with_same_sigma(data)
            else:
                self.plot_GDA_with_different_sigma(data, name="_with_linear_line")


    def plot_GDA_with_same_sigma(self, data):
        fig = plt.figure(figsize=(12, 8))
        foot = fig.add_subplot(1,1,1)

        if self.plot_on_same:
            plt.axis([-20, 20, -30, 30])


        foot.scatter(data.partition[0]['x'], data.partition[0]['y'], label="y = Canada | Data-label", color= "red", marker= ".")
        foot.scatter(data.partition[1]['x'], data.partition[1]['y'], label="y = Alaska | Data-label", color= "green", marker= "+")

        if self.plot_on_same:
            x=np.linspace(-30, 30, 500)
        else:
            x=np.linspace(min(min(data.partition[0]['x']), min(data.partition[1]['x'])), max(max(data.partition[0]['x']), max(data.partition[1]['x'])), data.samples)


        sigma_inverse = np.linalg.inv(data.sigma)
        theta = np.array([0, 0, 0])
        # Calculate theta: theta = [intercept + coefficients]
        # intercept:
        theta[0] = (data.mu1.T @ (sigma_inverse @ data.mu1)) - (data.mu0.T @ (sigma_inverse @ data.mu0)) - 2*log(data.phi/(1-data.phi))

        # for rest of the thetas
        # coefficients:
        coeffs = (sigma_inverse @ (data.mu0 - data.mu1)) + ((data.mu0 - data.mu1).T @ sigma_inverse)
        for i in range(2):
            theta[i+1] = coeffs[i]

        # plot the usual separation line
        y = -1 * (theta[1]*x + theta[0])/theta[2]

        foot.plot(x, y, label="Separation Line", color ='blue')

        if not self.plot_on_same:
            plt.xlabel('x - axis')
            plt.ylabel('y - axis')
            # plt.ylim(bottom=0)

            plt.legend()
            plt.title('Q4(c) - GDA: Covariance1 = Covariance2 | Linear Separation Line')
        else:
            plt.xlabel('x - axis')
            plt.ylabel('y - axis')
            # plt.ylim(bottom=0)

            plt.legend()
            plt.title('Q4(c) - GDA: Covariance1 = Covariance2 | Linear & Quadratic Separation Line')

        if not self.plot_on_same:
            plt.savefig('./graphs/gda_same_covariance_linear_line.png')


    def plot_GDA_with_different_sigma(self, data, name=""):
        if not self.plot_on_same:
            fig = plt.figure(figsize=(12, 8))
            foot = fig.add_subplot(1,1,1)

            foot.scatter(data.partition[0]['x'], data.partition[0]['y'], label="y = Canada | Data-label", color= "red", marker= ".")
            foot.scatter(data.partition[1]['x'], data.partition[1]['y'], label="y = Alaska | Data-label", color= "green", marker= "+")

        if self.plot_on_same:
            x=np.linspace(-30, 30, 500)
            y=np.linspace(-30, 30, 500)
        else:
            x = np.linspace(min(min(data.partition[0]['x']), min(data.partition[1]['x'])), max(max(data.partition[0]['x']), max(data.partition[1]['x'])), data.samples)
            y = np.linspace(min(min(data.partition[0]['y']), min(data.partition[1]['y'])), max(max(data.partition[0]['y']), max(data.partition[1]['y'])), data.samples)

        
        sigma0_inverse = np.linalg.inv(data.sigma0)
        sigma1_inverse = np.linalg.inv(data.sigma1)

        x, y = np.meshgrid(x, y)
        XYvalues = []
        for x1, x2 in zip(np.ravel(x), np.ravel(y)):
            temp = ((np.array([x1, x2]) - data.mu1) @  (sigma1_inverse @  (np.array([x1, x2]) - data.mu1))) - ((np.array([x1, x2]) - data.mu0) @  (sigma0_inverse @  (np.array([x1, x2]) - data.mu0))) + log(abs(np.linalg.det(data.sigma1)/np.linalg.det(data.sigma0))) + 2*log((1-data.phi)/(data.phi))
            XYvalues.append(temp)

        XYvalues = np.array(XYvalues).reshape(x.shape)
        contour = plt.contour(x, y, XYvalues, [0], colors='black')
        contour.collections[0].set_label("Quadratic Line")


        if not self.plot_on_same:
            plt.xlabel('x - axis | x1')
            plt.ylabel('y - axis | x2')
            # plt.ylim(bottom=0)

            plt.legend()
            plt.title('Q4(e) - GDA: Covariance1 != Covariance2 | Quadratic Separation Line')


        plt.savefig('./graphs/gda_different_covariance_quadratic_line{}.png'.format(name))



if __name__ == '__main__':
    import sys

    data = Datafiles(sys.argv[1], sys.argv[2])
    fig = Figures(data, int(sys.argv[3]), False)

