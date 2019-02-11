import csv
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np 


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
            X1.append(float(X[i][0]))
            Y1.append(float(Y[i][0]))

        X = X1
        Ylist = Y1

        # Normalize the data
        meanX = sum(X)/samples
        varianceX = 0
        for i in range(samples):
            varianceX += (X[i] - meanX)**2
        varianceX /= samples
        varianceX = math.sqrt(varianceX)

        for i in range(samples):
            X[i] = (X[i] - meanX)/varianceX

        # X values in list format
        Xlist = X

        # Make it into numpy array piecewise
        X1 = []
        for i in range(samples):
            X1.append(np.array([X[i]]))

        X = X1

        X = np.array(X)
        # Insert the X0 term (all ones: constant)
        X = np.insert(X, 0, 1.0, axis=1)
        Y = np.array(Ylist)

        self.X = X
        self.Y = Y 
        self.Xlist = Xlist
        self.Ylist = Ylist
        self.samples = samples

class LinearRegressionWithLocalWeights:
    def __init__(self, data, tau_list=[], draw_unweighted=True):
        if draw_unweighted:
            self.draw_unweighted_linear_regression(data)

        for t in tau_list:
            self.plot_theta_weighted(data, t)
    

    def plot_thetas(self, data, t):
        fig = plt.figure(figsize=(12, 6))
        foot = fig.add_subplot(1,1,1)

        foot.scatter(data.Xlist, data.Ylist, label="Training Data", color= "g", marker= ".")

        x=np.linspace(min(data.Xlist), max(data.Xlist), data.samples)
        y=list(t)[1]*x+list(t)[0]

        foot.plot(x, y, label="Hypothesis function learned", color ='b')

        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.ylim(bottom=0)

        plt.legend()
        plt.title('Q2(a) - Linear Regression (Unweighted): With Normal Equation')

        plt.savefig('./graphs/linear_regression_with_normal_equation.png')

    # Part (a)
    def draw_unweighted_linear_regression(self, data):
        theta = np.dot(np.dot(np.linalg.inv(np.dot(data.X.T, data.X)), np.transpose(data.X)), data.Y)
        # print("[*] Theta Values: ", theta)
        self.plot_thetas(data, theta)

    # Part (b)
    def plot_theta_weighted(self, data, tau):
        XYvalues = self.get_theta_weighted(data, tau)

        fig = plt.figure(figsize=(12, 6))
        foot = fig.add_subplot(1,1,1)

        foot.scatter(data.Xlist, data.Ylist, label="Training Data", color= "g", marker= ".")

        foot.plot(XYvalues[0], XYvalues[1], label="Hypothesis function learned",color ='b')

        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.ylim(bottom=0)

        plt.legend()
        plt.title('Q2(b) - Linear Regression Weighted (Tau = {})'.format(tau))

        plt.savefig('./graphs/linear_regression_weighted_tau{}.png'.format(tau))


    def get_theta_weighted(self, data, tau):
        x_val = np.linspace(min(data.Xlist), max(data.Xlist), data.samples)
        y_val = []
        
        for i in range(len(x_val)):
            weights = []
            for j in range(data.samples):
                temp = (x_val[i] - data.X[j][1])
                weights.append(math.exp((-1*temp*temp)/(2*tau*tau)))

            # Make diagonal matrix
            W = np.diag(np.array(weights))

            theta = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(data.X.T, W), data.X)), data.X.T), W), data.Y)
            y_val.append(theta[1]*x_val[i] + theta[0])

        return [x_val, y_val]

if __name__ == '__main__':
    data = Datafiles("./data/weightedX.csv", "./data/weightedY.csv")
    tau_list = [0.1, 0.3, 2, 10]
    lr = LinearRegressionWithLocalWeights(data, tau_list, True)