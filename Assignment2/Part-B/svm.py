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

		# Solving dual objective
		K = Y[:, None] * X 
		K = np.dot(K, K.T)

		K = K.astype(float)
		X = X.astype(float)
		Y = Y.astype(float)

		# print(K.shape)

		P = matrix(K)
		q = matrix(-np.ones((num_examples, 1)))

		G = matrix(-np.eye(num_examples))
		h = matrix(np.zeros(num_examples))
		
		A = matrix(Y.reshape(1, -1))
		b = matrix(np.zeros(1))
		
		if self.verbose:
			print("\n[!] Solving by CSXOPT!")

		solvers.options['show_progress'] = self.verbose

		sol = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(sol['x'])
		
		return alphas

	def weights_and_bias(self, alphas, data):
		X = np.array(data["data"])
		Y = np.array(data["label"])

		# get weights
		weights = np.sum(alphas * Y[:, None] * X, axis=0)
		# get bias
		condition = (alphas > 1e-4).reshape(-1)
		bias = Y[condition] - np.dot(X[condition], weights)

		weights = weights.reshape((weights.shape[0], 1))
		# print(alphas.shape)
		# print(weights.shape)
		# print(weights)
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
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))
	print("[*] Test Error Rate: {}\n".format(1-accuracy))


if __name__ == '__main__':
	main()