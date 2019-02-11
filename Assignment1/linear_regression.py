# Implementing Linear Regression: It takes time to do this too.
# Start: 3:34am
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def points2D(X, y, theta):
	plt.figure(figsize=(10,6))
	plt.scatter(X, y)

	# ax.plot(x, fit[0] * x + theta[])

	plt.savefig('points.png')


class LinearRegression:
	def __init__(self, X, y):
		self.X = X
		self.y = y
		
		self.max_iterations = 10
		self.alpha = 0.0001

		self.m = X.shape[0] # no of examples
		self.n = X.shape[1] # no of feature - here one

		t = (self.gradient_descent())
		print(t[0])

	def cost_function(self, theta):
		"""get cost function"""
		return ((theta.dot(self.X.T)-self.y)**2)/(self.m * 2)

	def gradient_descent(self):
		theta = np.zeros(shape=(2, 1)) # column vector theta (zero initialisation)
		self.X = np.insert(self.X, 0, 1.0, axis=1)
		print('\n[*] Init theta values ', theta)

		iteration = 0
		while True:
			theta = theta - self.alpha*((self.X.T.dot(self.X.dot(theta) - self.y)).sum())/self.m

			iteration += 1
			if iteration >= self.max_iterations:
				break

		return theta


if __name__ == '__main__':
	x = pd.read_csv("./data/linearX.csv", 'r')
	x = x.as_matrix()

	# numpy function asmatrix() doesn't copy data look into that
	y = pd.read_csv("./data/linearY.csv", 'r')
	y = y.as_matrix().flatten()

	# points2D(x, y)

	lr = LinearRegression(x, y)