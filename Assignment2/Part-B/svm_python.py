# SVM Model
# add libsvm in path
import sys
sys.path.append('/home/student_bharti/Ankit/Machine-Learning-COL774/Assignment2/Part-B/libsvm/python')

import timeit
import numpy as np 

from svmutil import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from process import Processing


class SVM:
	def __init__(self, verbose=False, kernel_type='linear', gamma = 0.05, C = 1.0):
		self.verbose = verbose
		self.gamma = gamma
		self.C = C
		self.kernel_type = kernel_type

		if self.verbose:
			if self.kernel_type is 'linear':
				print("[!] SVM -> Kernel: {} | C: {}".format(self.kernel_type, self.C))

			elif self.kernel_type is 'gaussian':
				print("[!] SVM -> Kernel: {} | Gamma: {} | C: {}\n".format(self.kernel_type, self.gamma, self.C))

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

		start_time = timeit.default_timer()
		sol = solvers.qp(P, q, G, h, A, b)
		end_time = timeit.default_timer()

		if sol['status'] is "unknown":
			print("[*] Unknown solution returned")

		self.alphas = np.array(sol['x']).squeeze()
		
		# calculate weights and bias implicitly
		self.weights_and_bias(data)

		self.time_taken = end_time - start_time

	def weights_and_bias(self, data):
		X = np.array(data["data"])
		Y = np.array(data["label"])

		SV_idxs = list(filter(lambda i:self.alphas[i] > 1e-6, range(len(Y))))

		self.SV_X, self.SV_Y, self.alphas = X[SV_idxs], Y[SV_idxs], self.alphas[SV_idxs]
		self.SV_len = len(SV_idxs)

		if self.verbose:
			print("[*] {} number of support vectors!".format(self.SV_len))

		# get weights
		if self.kernel_type is 'linear':
			weights = np.dot(self.alphas * self.SV_Y, self.SV_X)
		
		SV_bound = self.alphas < self.C - 1e-6

		temp = np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, self.SV_X[SV_bound]))
		self.bias = np.mean(self.SV_Y[SV_bound] - temp)

		self.X = X
		self.Y = Y

		if self.kernel_type is 'linear':
			self.weights = weights

	def predict(self, test):
		predicted_labels = []
		for X in test["data"]:
			X = np.array(X)
			X = X.astype(float)
			X = X[:,None]
			pred = np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, X.T)) + self.bias

			if pred>0:
				predicted_labels.append(1)
			else:
				predicted_labels.append(2)	
		
		return predicted_labels

	def predict_score(self, X):
		"""predict a singleton test"""

		pred = np.dot((self.alphas * self.SV_Y), self.kernel(self.SV_X, X)) + self.bias
		return np.sign(pred)


class SVM_libsvm:
	def __init__(self, kernel_type='gaussian', C=1.0, gamma=0.05):
		self.kernel_type = kernel_type
		self.C = C
		self.gamma = gamma

		print("[!] Kernel: {} | Gamma: {} | C: {}".format(self.kernel_type, self.gamma, self.C))


	def fit(self, data):
		prob = svm_problem(data["label"], data["data"])
		param = svm_parameter()
		
		if self.kernel_type is 'linear':
			param.kernel_type = LINEAR
		elif self.kernel_type is 'gaussian':
			param.kernel_type = RBF

		param.C = self.C
		param.gamma = self.gamma
		param.q = True

		start_time = timeit.default_timer()
		model = svm_train(prob, param)
		end_time = timeit.default_timer()
		
		self.model = model
		self.time_taken = end_time - start_time

	def predict(self, testdata, label1, label2):
		labels =  svm_predict(testdata["label"], testdata["data"], self.model)
		predicted_labels = []

		for l in labels[0]:
			if l>0:
				predicted_labels.append(label1)
			else:
				predicted_labels.append(label2)	
		
		return predicted_labels

	def predict_score(self, testdata):
		labels =  svm_predict(testdata["label"], testdata["data"], self.model)
		return np.array(labels[0])


# main function for svm binary classification
def main():
	# processing for training
	p = Processing(train_file="./dataset/train.csv", test_file="./dataset/test.csv")
	# p.process_data()

	# create model
	s = SVM(verbose=True, kernel_type='linear')
	s.fit(p.data)

	print("[*] Bias: {}".format(str(s.bias)))	
	# make prediction
	predicted_labels = s.predict(p.testdata)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	print('[!] Computational time (of training): {}'.format(s.time_taken))

	# print("[*] Support Vectors: \n", s.SV_X[0])

# use LIBSVM
def main_use_libsvm():
	p = Processing(train_file="./dataset/train.csv", test_file="./dataset/test.csv")

	s = SVM_libsvm(kernel_type='gaussian', C=1.0, gamma=0.05)
	s.fit(p.data)

	predicted_labels = s.predict(p.testdata, p.label1, p.label2)
	# print(predicted_labels)
	# print(p.testdata["label"])

	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("\n[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	print('[!] Computational time (of training): {}'.format(s.time_taken))

if __name__ == '__main__':
	main_use_libsvm()
	# main()