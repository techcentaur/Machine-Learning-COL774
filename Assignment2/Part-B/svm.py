# SVM Model

import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from process import Processing

class SVM:
	def __init__(self, verbose=False, kernel_type='linear'):
		self.verbose = verbose
		self.gamma = 0.05
		self.noise = 1.0
		self.kernel_type = kernel_type

	def kernel(self, x, x_dash=None):
		# one-arg -> x_dash = x
		if x_dash is None:
			x_dash = x
	
		if self.kernel_type is 'linear':
			return np.dot(x, x_dash.T)
		elif self.kernel_type is 'gaussian':
			x_sq = np.sum(x*x, axis=1)
			x_dash_sq = np.sum(x_dash*x_dash, axis=1)

			val = x_sq.reshape((-1, 1)) + x_dash_sq.reshape((1, -1)) - 2.0 * np.dot(x, x_dash.T)
			# print(np.exp(-self.gamma * val))
			return np.exp(-self.gamma * val)
		else:
			print("[!] No known kernel type!")
			return False

	def fit(self, data):		# Fit according to QP formulation (dual problem)
		# m variables and 2m+1 constraints
		num_examples = len(data["data"]) # m (examples)
		num_features = len(data["data"][0]) # n (features)

		if self.verbose:
			print("[!] Number of examples {} and number of features {}".format(num_examples, num_features))

		X = np.array(data["data"])
		Y = np.array(data["label"])

		# Solving dual objective
	
		K = self.kernel(X) * np.outer(Y, Y)
		# X_dash= X_dash.reshape((X.shape[0]), 1)
		# K = K.astype(float)
		K = K/2
		print(K)
		P = matrix(K, tc='d')
		q = matrix(-1*np.ones((num_examples, 1)), tc='d')

		G = matrix(np.r_[np.eye(num_examples), -1*np.eye(num_examples)], tc='d')
		h = matrix(np.r_[np.ones((num_examples, 1)), np.zeros((num_examples, 1))], tc='d')
		
		A = matrix(Y.reshape(1, -1), tc='d')
		b = matrix([0.0], tc='d')
		
		# print(P.size, q.size, G.size, h.size, A.size, b.size)

		if self.verbose:
			print("\n[!] Solving by CSXOPT!")

		solvers.options['show_progress'] = self.verbose

		sol = solvers.qp(P, q, G, h, A, b)

		if sol['status'] is "unknown":
			print("[*] Unknown solution returned")

		alphas = np.array(sol['x']).squeeze()
		print(alphas)
		return alphas

	def weights_and_bias(self, alphas, data):
		X = np.array(data["data"])
		Y = np.array(data["label"])

		SV_idxs = list(filter(lambda i:alphas[i] > 1e-6, range(len(Y))))

		self.SV_X, self.SV_Y, self.alphas = X[SV_idxs], Y[SV_idxs], alphas[SV_idxs]
		self.SV_len = len(SV_idxs)

		if self.verbose:
			print("[*] {} number of support vectors!", self.SV_len)

		# get weights
		if self.kernel_type is 'linear':
			weights = np.dot(self.alphas * self.SV_Y, self.SV_X)
		
		SV_bound = self.alphas < self.noise - 1e-6

		temp = np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, self.SV_X[SV_bound]))
		bias = np.mean(self.SV_Y[SV_bound] - temp)

		self.X = X
		self.Y = Y

		if self.kernel_type is 'linear':
			self.weights = weights

		self.alphas = alphas
		self.bias = bias 

	def predict(self, test):
		predicted_labels = []
		for X in test["data"]:
			X = np.array(X)
			X = X.reshape((-1, 1))
			X = X.astype(float)

			pred = np.dot(self.alphas * self.SV_Y, self.kernel(self.SV_X, X.T)) + self.bias
			# pred = np.dot(self.SV_Y * self.alphas.T, self.kernel(self.SV_X, X)) + self.bias

			predicted_labels.append(1 if pred[0] > 0 else 2)
		
		# print(predicted_labels)
		return predicted_labels
		# return [1 if x>0 else 2 for x in predicted_labels]


def main():
	# processing for training
	p = Processing(train_file="./dataset/train.csv", test_file="./dataset/test.csv")
	p.process_data()

	# create model
	s = SVM(verbose=True)
	alphas = s.fit(p.data)

	# find weights and bias
	s.weights_and_bias(alphas, data["train"])

	# make prediction
	predicted_labels = s.predict(data["test"])
	print(predicted_labels)
	print(data["test"]["label"])

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == data["test"]["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	# print("[*] Support Vectors: \n", s.SV_X[0])

if __name__ == '__main__':
	main()