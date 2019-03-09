# SVM Model

import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from process import Processing

class SVM:
	def __init__(self):
		self.verbose = True

	def fit(self, data):
		num_examples = len(data["data"])
		num_features = len(data["data"][0])

		if self.verbose:
			print("[!] Number of examples {} and number of features {}".format(num_examples, num_features))



		# Solving dual objective


		# K = np.dot(K, K.T)
		# P = matrix(K)
		# q = matrix(-np.ones((NUM, 1)))
		# G = matrix(-np.eye(NUM))
		# h = matrix(np.zeros(NUM))
		# A = matrix(y.reshape(1, -1))
		# b = matrix(np.zeros(1))
		# solvers.options['show_progress'] = False
		# sol = solvers.qp(P, q, G, h, A, b)
		# alphas = np.array(sol['x'])
		# return alphas


if __name__ == '__main__':
	p = Processing(train_file="./dataset/train.csv")
	p.process_data()

	s = SVM()
	s.fit(p.data)
