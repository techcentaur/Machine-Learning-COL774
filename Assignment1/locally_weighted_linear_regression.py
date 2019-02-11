import csv
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np 

# Read from CSV file
csvfileX=open("./data/weightedX.csv", 'r')
csvfileY=open("./data/weightedY.csv", 'r')

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

# ---------------------- File Reading End----------------------------------------


def plot_thetas(Xlist, Ylist, t):
    fig = plt.figure(figsize=(12, 6))
    foot = fig.add_subplot(1,1,1)

    foot.scatter(Xlist, Ylist, label="Training Data", color= "g", marker= ".")

    x=np.linspace(min(Xlist), max(Xlist), samples)
    y=list(t)[1]*x+list(t)[0]

    foot.plot(x, y, label="Hypothesis function learned",color ='b')

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.ylim(bottom=0)

    plt.legend()
    plt.title('Q2(a) - Linear Regression (Unweighted): With Normal Equation')

    plt.savefig('linear_regression_with_normal_equation.png')


# Part (a)

theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), np.transpose(X)), Y)
# print("[*] Theta Values: ", theta)

plot_thetas(Xlist, Ylist, theta)


# Part (b)

def plot_theta_weighted(Xlist, Ylist, tau):
    XYvalues = get_theta_weighted(Xlist, Ylist, tau)

    fig = plt.figure(figsize=(12, 6))
    foot = fig.add_subplot(1,1,1)

    foot.scatter(Xlist, Ylist, label="Training Data", color= "g", marker= ".")

    foot.plot(XYvalues[0], XYvalues[1], label="Hypothesis function learned",color ='b')

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.ylim(bottom=0)

    plt.legend()
    plt.title('Q2(a) - Linear Regression Weighted (Tau = {})'.format(tau))

    plt.savefig('linear_regression_weighted_tau{}.png'.format(tau))


def get_theta_weighted(Xlist, Ylist, tau):
    x_val = np.linspace(min(Xlist), max(Xlist), samples)
    y_val = []
    
    for i in range(len(x_val)):
        weights = []
        for j in range(samples):
            temp = (x_val[i] - X[j][1])
            weights.append(math.exp((-1*temp*temp)/(2*tau*tau)))

        # Make diagonal matrix
        W = np.diag(np.array(weights))

        theta = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), X.T), W), Y)
        y_val.append(theta[1]*x_val[i] + theta[0])

    return [x_val, y_val]


tau_values = [0.1, 0.3, 2, 10]
for t in tau_values:
    plot_theta_weighted(Xlist, Ylist, t)
