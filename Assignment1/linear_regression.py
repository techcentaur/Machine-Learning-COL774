import os
import csv
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np 
import imageio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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


def plot_thetas(data, t, rate):
    fig = plt.figure(figsize=(12, 6))
    foot = fig.add_subplot(1,1,1)

    foot.scatter(data.Xlist, data.Ylist, label="Training Data", color= "g", marker= ".")

    x = np.linspace(min(data.Xlist), max(data.Xlist), data.samples)
    y = list(t)[1]*x+list(t)[0]

    foot.plot(x, y, label="Hypothesis function learned", color ='b')

    plt.xlabel('x - axis | Acidity')
    plt.ylabel('y - axis | Density')

    plt.legend()
    plt.title('Q1(a) - Linear Regression - Hypothesis Function Learned')

    plt.savefig('./graphs/linear_regression_hypothesis_function_plot_{}.png'.format(rate))

    print("[*] Linear regression hypothesis function saved.\n")


def plot_error_function_in_gif(data, t, it, axis, rate):
    if not os.path.exists("./graphs/error_function_{}".format(rate)):
        os.makedirs("./graphs/error_function_{}".format(rate))

    axis.scatter(t[0], t[1], np.sum(((data.Y - (data.X @ t))**2)/(2*data.samples)))

    plt.xlabel('theta-0')
    plt.ylabel('theta-1')
    # plt.ylim(bottom=0)

    plt.legend()

    plt.savefig('./graphs/error_function_{r}/{i}.png'.format(r=rate, i=it))

def plot_contour_values(x1, x2, function_values, cost_theta, rate):
    if not os.path.exists("./graphs/contour_values_{}".format(rate)):
        os.makedirs("./graphs/contour_values_{}".format(rate))

    fig = plt.figure(figsize=(12, 8))
    foot = fig.add_subplot(1,1,1)

    plt.xlabel('x - axis | theta-0')
    plt.ylabel('y - axis | theta-1')
    plt.title('Contour graph: Eta value {}'.format(rate))
    # plt.ylim(bottom=0)

    for i in range(len(cost_theta)):
        plt.contour(x1, x2, function_values, 50)
        plt.scatter(cost_theta[i][1][0], cost_theta[i][1][1])
        plt.savefig('./graphs/contour_values_{l}/{i}.png'.format(l=rate, i=i))

def plot_contour_values_plot_once(x1, x2, function_values, cost_theta, rate):
    fig = plt.figure(figsize=(12, 8))
    foot = fig.add_subplot(1,1,1)

    plt.xlabel('x - axis | theta-0')
    plt.ylabel('y - axis | theta-1')
    plt.title('Contour graph: Eta value {}'.format(rate))
    # plt.ylim(bottom=0)

    for i in range(len(cost_theta)):
        plt.contour(x1, x2, function_values, 50)
        plt.scatter(cost_theta[i][1][0], cost_theta[i][1][1])

    plt.savefig('./graphs/contour_values_eta_{}.png'.format(rate))


class LinearRegression:
    def __init__(self, data, rate):
        self.threshold = 1e-12
        self.learning_rate = rate
        self.batch_gradient_descent(data)
        self.error_function_with_params(data)

    def cost(self, t):
        return np.sum(((data.Y - (data.X @ t))**2)/(2*data.samples))

    def batch_gradient_descent(self, data):
        theta = np.array([0, 0])

        i = 0
        while True:
            # calculate htheta
            htheta = (data.X @ theta)
            # calculate gradient
            gradient = ((data.Y - htheta) @ data.X)/(data.samples)
            theta_old = theta
            
            # update theta
            theta = theta + self.learning_rate * gradient

            if(abs(self.cost(theta_old) - self.cost(theta)) < self.threshold):
                print("{:=^70}\n".format(' Part(a) '))
                print("\n[.]------------------ Convergence Criteria :----------------")
                print("[!] Converged as change in theta is too low [!]")
                print("[*] Change in theta: {i1} < {i2} (set threshold)".format(i1=abs(self.cost(theta_old) - self.cost(theta)), i2=self.threshold))
                print("[*] At iteration: {}\n".format(i))
                break
            i+=1

        print("\n[.]--------------------- Model Parameters ------------------")
        print("[*] Learning Rate: {}".format(self.learning_rate))
        print("[*] Thetas: {}".format(theta))
        print("[*] Final Cost: {}".format(self.cost(theta)))


        plot_thetas(data, theta, self.learning_rate)
        print("{:=^70}\n".format(' Part(c+) '))
        self.theta = theta


    def error_function_with_params(self, data):
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-1, 1, 100)
        x1, x2 = np.meshgrid(x1, x2)

        function_values = []
        for _x1, _x2 in zip(np.ravel(x1), np.ravel(x2)):
            temp = self.cost(np.array([_x1, _x2]))
            function_values.append(temp)

        function_values = np.array(function_values).reshape(x1.shape)

        fig = plt.figure(figsize=(8,6))
        axis = fig.add_subplot(111, projection='3d')
        surface = axis.plot_surface(x1, x2, function_values, cmap=cm.coolwarm, alpha=0.5, label="Error function")

        surface._facecolors2d=surface._facecolors3d
        surface._edgecolors2d=surface._edgecolors3d

        plt.title('Q1(c) - Linear Regression: Error (J) vs Parameters')
        
        cost_theta = []
        theta = np.array([0, 0])
        cost_theta.append((self.cost(theta), theta))
        
        i = 0
        
        while True:
            htheta = (data.X @ theta)
            gradient = ((data.Y - htheta) @ data.X)/data.samples
            
            theta_old = theta
            theta = theta + self.learning_rate * gradient
            
            cost_theta.append((self.cost(theta), theta))
            
            plot_error_function_in_gif(data, theta, i, axis, self.learning_rate)
            if(abs(self.cost(theta_old) - self.cost(theta)) < self.threshold):
                break
            i+=1

        plot_contour_values(x1, x2, function_values, cost_theta, self.learning_rate)
        # plot_contour_values_plot_once(x1, x2, function_values, cost_theta, self.learning_rate)

def create_animation(rootdir, delay):
    images,image_file_names = [],[]
    for file_name in os.listdir(rootdir):
        if file_name.endswith('.png'):
            image_file_names.append(file_name)

    sorted_files = sorted(image_file_names, key=lambda y: int(y.split('.')[0]))

    # define some GIF parameters
    
    frame_length = delay # seconds between frames
    end_pause = 4 # seconds to stay on last frame
    # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
    for ii in range(0,len(sorted_files)):       
        file_path = os.path.join(rootdir, sorted_files[ii])
        if ii==len(sorted_files)-1:
            for jj in range(0,int(end_pause/frame_length)):
                images.append(imageio.imread(file_path))
        else:
            images.append(imageio.imread(file_path))
    # the duration is the time spent on each image (1/duration is frame rate)
    n = rootdir.split("/")[-1]
    print("[*] Saving animation for {}".format(n))
    imageio.mimsave("./graphs/{}.gif".format(n), images,'GIF',duration=frame_length)



if __name__ == '__main__':
    import sys
    data = Datafiles(sys.argv[1], sys.argv[2])
    # for rate in [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]:
    lr = LinearRegression(data, float(sys.argv[3]))

    # create_animation("./graphs/linear_theta_lines", float(sys.argv[4]))
    create_animation("./graphs/error_function_{}".format(sys.argv[3]), float(sys.argv[4]))
    create_animation("./graphs/contour_values_{}".format(sys.argv[3]), float(sys.argv[4]))

    print("[*] Part of the journey is the end!")
