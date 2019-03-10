# SVM Model

import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from process import Processing

class SVM:
	def __init__(self, verbose=False):
		self.verbose = verbose

	def fit(self, data):
		num_examples = len(data["data"])
		num_features = len(data["data"][0])

		if self.verbose:
			print("[!] Number of examples {} and number of features {}".format(num_examples, num_features))

		X = np.array(data["data"])
		Y = np.array(data["label"])

		print(Y.shape)
		print(X.shape)
		# print(Y[:, None].shape)
		# print((Y.reshape(1, -1)).shape)

		# Solving dual objective
		# Y_dash = Y[:, None] @ Y.reshape(1, -1)
		Y_dash = np.outer(Y[:, None], Y[:, None])
		X_dash = np.dot(X, X.T)
		print(X_dash.shape)
		print(Y_dash.shape)
		 
		# X_dash= X_dash.reshape((X.shape[0]), 1)
		# print(K.shape)
		K = (Y_dash * X_dash)
		print(K.shape)

		K = K.astype(float)
		X = X.astype(float)
		Y = Y.astype(float)

		# print(K.shape)
		# print(K)

		P = matrix(K)
		q = matrix(-np.ones((num_examples, 1)))

		G = matrix(-np.eye(num_examples))
		h = matrix(np.zeros(num_examples))
		
		A = matrix(Y.reshape(1, -1))
		b = matrix([0.0])
		
		print(P.size, q.size, G.size, h.size, A.size, b.size)

		if self.verbose:
			print("\n[!] Solving by CSXOPT!")

		solvers.options['show_progress'] = self.verbose

		sol = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(sol['x']).squeeze()
		
		return alphas

	def weights_and_bias(self, alphas, data):
		X = np.array(data["data"])
		Y = np.array(data["label"])

		# get weights
		weights = np.dot(alphas * Y[:, None], X)
		# get bias
		# condition = (alphas > 1e-4).reshape(-1)
		print(weights.shape)
		# bias = np.mean(Y - np.dot(X, weights))

		# weights = weights.reshape((weights.shape[0], 1))
		print(alphas)
		print(weights)
		# print(bias)
		bias = 0
		self.weights = weights
		self.bias = bias 

	def predict(self, test):
		predicted_labels = []
		for X in test["data"]:
			predicted_labels.append((self.weights.T @ X) + self.bias)
		
		return [1 if x>0 else 2 for x in predicted_labels]


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
	# print(predicted_labels)
	# print(data["test"]["label"])

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == data["test"]["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))


if __name__ == '__main__':
	main()