import numpy as np
import pandas as pd

class LinearRegression:
	def __init__(self, params, X):
		self.max_iterations = 1000
		self.alpha = 0.0001
		self.m = x.shape[0]

	def initialize_thetas(self):
		self.theta = [0, 0]

	def cost_function(self):
		cost = (0.5*(sum([(self.theta[0] + self.theta[1]*np.asarray([x[i]]) - y[i])**2 for i in range(self.m)])))/self.m
		return cost

	def gradient_descent(self):
		self.initialize_thetas()

		iteration = 0
		converged = False
		while not converged:
			temp0 = sum([(self.theta[0] + self.theta[1]*np.asarray([x[i]]) - y[i]) for i in range(self.m)])/self.m
			temp1 = sum([(self.theta[0] + self.theta[1]*np.asarray([x[i]]) - y[i])*np.asarray(x[i]) for i in range(m)])/self.m

			self.theta[0] = self.theta[0] - self.alpha*temp0
			self.theta[1] = self.theta[1] - self.alpha*temp1

			iteration += 1

			if iteration >= self.max_iterations:
				converged = True

		return self.theta


if __name__ == '__main__':
	params = {
		"max_iterations": 1000,
		"alpha": 0.01, #learning rate
		"epsilon": 0.01 #convergence criteria
	}

	df = pd.read_csv('linearX.csv', names=['x'])- 














	df = pd.read_csv('linearY.csv', names=['y'])
	
	X = df['x']
	y = df['y']

	lr = LinearRegression(params, X, y)

