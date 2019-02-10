import numpy as np 
import pandas as pd


class LogisticRegression:
	def __init__(self, X, y):
		self.X = np.insert(X, 0, 1.0, axis=1) #insert X0 for theta0
		self.y = y

		self.max_iter = 5
		self.threshold = 1.0e-2

		self.m = X.shape[0]  #samples
		self.n = X.shape[1]-1 #features

	def initialise_thetas(self):
		return np.zeros([self.n+1, ])  #extra theta0 term

	def sigmoid(z):
		"""return the sigmoid value of z(can be a matrix)"""
		return 1/(1 + np.exp(-z))


	def get_gradient(self, thetas):
		"""get gradient of log likelihood"""

		gradient = np.zeros([self.n+1, ])

		# i - ith example; j - jth feature
		for j in range(self.n+1):
			_grad = 0.0
			for i  in range(0, self.m):
				_grad += (self.y[i] - self.sigmoid(thetas @ self.X[i])).flatten()[0] * self.X[i][j]
			gradient[j] = _grad

		return gradient

	def get_hessian(self, thetas):
		"""get hessian matrix from thetas of loglikelihood"""
		hessian = np.zeros([self.n+1, self.n+1])

		# i - ith feature, j - jth feature, k - kth example
		for i in range(self.n+1):
			for j in range(self.n+1):
				hessian[i][j] = 0.0
				for k in range(self.m):
					sig = self.sigmoid(thetas @ self.X[k])
					hessian[i][j] += self.X[m][i]*self.X[m][j]*sig*(sig-1)

		return hessian


	def newtons_method(self):
		# get theta
		thetas = self.initialise_thetas() 

		# do iterations
		iteration = 0
		converged = False
		while not converged:
			grad = self.get_gradient(thetas)
			loss_val = np.abs(grad)[np.argmax(grad)]

			print(iteration, " ", thetas, " ", loss_val)

			if(loss_val < self.threshold or iteration == self.max_iter):
				converged = True
				continue

			# get hessian and calculate its inverse
			hessian = np.matrix(self.get_hessian(thetas))

			# update theta as: theta = theta - hessian_inverse * grad
			try:
				thetas = thetas - np.array( hessian.I @ grad)
			except Exception as e:
				print("[!] Hessian matrix not invertible")

			# thetas.shape = [self.n+1, ]
			iteration += 1

		# self.save_parameters(thetas)

	def save_parameters(self, thetas):
		with open('param.txt', 'w') as f:
			f.write(thetas)

		print("[*] Parameters Saved!")



class ReadFiles:
	def __init__(self):
		pass

	def read_file(self, filenameX, filenameY):
		x = pd.read_csv(filenameX, 'r')
		x = x.as_matrix()
		# numpy function asmatrix() doesn't copy data look into that
		y = pd.read_csv(filenameY, 'r')
		y = y.as_matrix().flatten()
		
		return {"input": x, "target": y}

if __name__ == '__main__':
	r = ReadFiles()
	df = r.read_file("./data/logisticX.csv", "./data/logisticY.csv")
	# print(df)

	# print(df["input"].shape)
	# print(df["target"].shape)
	lr = LogisticRegression(df["input"], df["target"])
	lr.newtons_method()