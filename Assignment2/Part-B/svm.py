# SVM Model

import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from process import Processing

class SVM:
	def __init__(self, verbose=False, kernel_type='gaussian'):
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
			# print(x.shape)
			# print(x_dash.shape)
			x_sq = np.sum(x*x, axis=1)
			x_dash_sq = np.sum(x_dash*x_dash, axis=1)
			# print(x_sq.shape)
			# print(x_dash_sq.shape)
			# print(x_sq.reshape((-1, 1)).shape)
			# print(x_dash_sq.reshape((1, -1)).shape)
			# print("----------------------")

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
		# Y = Y[:, None]
	
		K = self.kernel(X) * np.outer(Y, Y)
		# print(K.shape)
		 
		# X_dash= X_dash.reshape((X.shape[0]), 1)

		# K = K.astype(float)
	
		P = matrix(K, tc='d')
		q = matrix(-np.ones((num_examples, 1)), tc='d')

		G = matrix(np.r_[-np.eye(num_examples), np.eye(num_examples)], tc='d')
		h = matrix(np.r_[np.zeros((num_examples, 1)), np.zeros((num_examples, 1)) + self.noise], tc='d')
		
		A = matrix(Y.reshape(1, -1), tc='d')
		b = matrix([0.0])
		
		# print(P.size, q.size, G.size, h.size, A.size, b.size)

		if self.verbose:
			print("\n[!] Solving by CSXOPT!")

		solvers.options['show_progress'] = self.verbose

		sol = solvers.qp(P, q, G, h, A, b)
		if sol['status'] is "unknown":
			print("[*] Unknown solution returned")

		alphas = np.array(sol['x']).squeeze()
		
		return alphas

	def weights_and_bias(self, alphas, data):
		X = np.array(data["data"])
		Y = np.array(data["label"])

		SV_idxs = list(filter(lambda i:alphas[i] > 0, range(len(Y))))
		# print(SV_idxs)
		self.SV_X, self.SV_Y, self.alphas = X[SV_idxs], Y[SV_idxs], alphas[SV_idxs]
		self.SV_len = len(SV_idxs)

		# get weights
		if self.kernel_type is 'linear':
			weights = np.dot(self.alphas * self.SV_Y, self.SV_X)
		
		SV_bound = self.alphas < self.noise - 1e-6
		# print(SV_bound)
		# bias = np.mean(self.SV_Y[SV_bound] - np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, self.SV_X[SV_bound]))
		temp = np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, self.SV_X[SV_bound]))
		bias = np.mean(self.SV_Y[SV_bound] - temp)
		# get bias
		# condition = (alphas > 1e-4).reshape(-1)
		# print(weights)
		# bias = np.mean(Y - np.dot(X, weights))

		# weights = weights.reshape((weights.shape[0], 1))
		# print(bias)
		# bias = 0
		self.X = X
		self.Y = Y
		if self.kernel_type is 'linear':
			self.weights = weights

		# self.alphas = alphas
		self.bias = bias 

	def predict(self, test):
		predicted_labels = []
		for X in test["data"]:
			X = np.array(X)
			X = X.reshape((-1, 1))
			X = X.astype(float)

			pred = np.dot(self.alphas * self.SV_Y, self.kernel(self.SV_X, X.T)) + self.bias
			# pred = np.dot(self.SV_Y * self.alphas.T, self.kernel(self.SV_X, X)) + self.bias
			# print(pred[0])
			predicted_labels.append(1 if pred[0] > 1 else 0)
			# np.sign
		
		# print(predicted_labels)
		return predicted_labels
		# return [1 if x>0 else 2 for x in predicted_labels]


def main():
	# do processing
	p = Processing(train_file="./dataset/train.csv")
	p.process_data()
	data = p.train_and_test()

	# create model
	s = SVM(verbose=False)
	alphas = s.fit(data["train"])

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