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

		# print(len(weights))
		print(alphas.shape)
		print(weights.shape)
		print(bias)
		bias = bias[0]


def main():
	# do processing
	p = Processing(train_file="./dataset/train.csv")
	p.process_data()

	# create model
	s = SVM(verbose=False)
	alphas = s.fit(p.data)

	# find weights and bias
	s.weights_and_bias(alphas, p.data)


if __name__ == '__main__':
	main()