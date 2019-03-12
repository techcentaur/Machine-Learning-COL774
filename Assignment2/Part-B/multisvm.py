"""Multi-class classification SVM"""
import sys
sys.path.append('/home/student_bharti/Ankit/Machine-Learning-COL774/Assignment2/Part-B/libsvm/python')

from svmutil import *

import os
import copy
import timeit
import numpy as np 
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from sklearn.metrics import confusion_matrix
import seaborn as sns

from svm_python import (SVM, SVM_libsvm)
from process import (Processing, ProcessingForMulti)


class MultiSVM:
	def __init__(self, verbose=False, kernel_type='gaussian', gamma = 0.05, C = 1.0):
		# parameters
		self.gamma = gamma
		self.C = C
		self.kernel_type = kernel_type
		self.verbose = verbose

		self.num_classes = 0
		self.binary_svm_instances = []

		if self.verbose:
			if self.kernel_type is 'linear':
				print("[!] MultiSVM -> Kernel: {} | C: {}".format(self.kernel_type, self.C))

			elif self.kernel_type is 'gaussian':
				print("[!] MultiSVM -> Kernel: {} | Gamma: {} | C: {}\n".format(self.kernel_type, self.gamma, self.C))


	def fit(self, data):
		unique_labels = np.unique(data["label"])
		self.num_classes = len(np.unique(data["label"]))
		
		start_time = timeit.default_timer()

		if self.verbose:
			print("[!] Training examples: {}".format(len(data["data"])))

		# one-vs-one classifier method
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# SVM classifier has one pos and one neg id
				negative = data["label"] == unique_labels[i]
				positive = data["label"] == unique_labels[j]

				X_data = np.r_[data["data"][negative], data["data"][positive]]
				Y_data = np.r_[data["label"][negative], data["label"][positive]]

				# divide the data into two categories
				Y_data[Y_data == unique_labels[i]] = -1.0
				Y_data[Y_data == unique_labels[j]] = 1.0

				if self.verbose:
					print("[*] Classifier: (i={}, j={})".format(i, j))


				svm_object = SVM(self.verbose, self.kernel_type, self.gamma, self.C)
				svm_object.fit({"data": X_data, "label": Y_data})

				self.binary_svm_instances.append(copy.deepcopy(svm_object))
		
		end_time = timeit.default_timer()
		self.time_taken = end_time - start_time
		

	def predict(self, testdata):
		num_test_data = len(testdata["data"])

		if self.verbose:
			print("[!] Testing examples: {}".format(num_test_data))

		svm_instances = (self.num_classes * (self.num_classes-1))/2
		predictions = np.zeros((num_test_data, self.num_classes))

		svm_object_id = 0
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# get predicted label for each class and report the maximum wins
				pred = self.binary_svm_instances[svm_object_id].predict_score(testdata["data"])
				# pred<0: label i
				predictions[pred<0, i] += 1.0
				# pred>0: label j
				predictions[pred>0, j] += 1.0
				svm_object_id += 1


		preds = []
		for p in predictions:
			temp = np.where(p==p.max())
			preds.append(temp[0][temp[0].argmax()])

		preds = np.array(preds)

		return preds



	def draw_confusion_matrix(self, actual_y, predicted_y, name=''):
		"""draw confusion matrix as the name suggest"""

		cm = confusion_matrix(actual_y, predicted_y)

		if not os.path.exists("./figures"):
			os.makedirs("./figures")

		fig, ax = plt.subplots(figsize=(18, 18))
		sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels');
		ax.set_ylabel('Actual labels');

		ax.set_title('Part-B | Confusion Matrix | Multi-Class CVXOPT | Test Data'); 
		ax.xaxis.set_ticklabels([str(i) for i in range(0, 10)]);
		ax.yaxis.set_ticklabels([str(i) for i in range(0, 10)]);

		plt.show(block=False)
		plt.savefig('./figures/confusion_matrix_{}.png'.format(name))

		print("[>] Saving confusion matrix as confusion_matrix_{}.png in the ./figures folder in the directory".format(name))



class MultiSVM_libsvm:
	def __init__(self, verbose=True, C = 1.0, kernel_type='gaussian'):
		# parameters
		self.gamma = 0.05
		self.C = C
		self.kernel_type = kernel_type
		self.verbose = verbose

		self.num_classes = 0
		self.binary_svm_instances = []


	def fit(self, data):
		unique_labels = np.unique(data["label"])
		self.num_classes = len(np.unique(data["label"]))

		# one-vs-one classifier method
		start_time = timeit.default_timer()

		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# SVM classifier has one pos and one neg id
				negative = data["label"] == unique_labels[i]
				positive = data["label"] == unique_labels[j]

				X_data = np.r_[data["data"][negative], data["data"][positive]]
				Y_data = np.r_[data["label"][negative], data["label"][positive]]

				# divide the data into two categories
				Y_data[Y_data == unique_labels[i]] = -1.0
				Y_data[Y_data == unique_labels[j]] = 1.0

				if self.verbose:
					print("[*] Classifier: (i={}, j={})".format(i, j))

				svm_object = SVM_libsvm(kernel_type=self.kernel_type, C=self.C)
				svm_object.fit({"data": X_data, "label": Y_data})

				self.binary_svm_instances.append(svm_object)

		end_time = timeit.default_timer()
		self.time_taken = end_time - start_time


	def predict(self, testdata):
		num_test_data = len(testdata["data"])

		# if self.verbose:
		print("[!] testing examples: {}".format(num_test_data))

		svm_instances = (self.num_classes * (self.num_classes-1))/2
		predictions = np.zeros((num_test_data, self.num_classes))

		svm_object_id = 0
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# get predicted label for each class and report the maximum wins
				pred = self.binary_svm_instances[svm_object_id].predict_score(testdata)
			
				# pred<0: label i
				predictions[pred<0, i] += 1.0
				# pred>0: label j
				predictions[pred>0, j] += 1.0
				svm_object_id += 1

		preds = []
		for p in predictions:
			temp = np.where(p==p.max())
			preds.append(temp[0][temp[0].argmax()])

		preds = np.array(preds)

		return preds


	def draw_confusion_matrix(self, actual_y, predicted_y, name=''):
		"""draw confusion matrix as the name suggest"""

		cm = confusion_matrix(actual_y, predicted_y)

		if not os.path.exists("./figures"):
			os.makedirs("./figures")

		fig, ax = plt.subplots(figsize=(18, 18))
		sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels');
		ax.set_ylabel('Actual labels');

		ax.set_title('Part-B | Confusion Matrix | Multi-Class LIBSVM | Test Data'); 
		ax.xaxis.set_ticklabels([str(i) for i in range(0, 10)]);
		ax.yaxis.set_ticklabels([str(i) for i in range(0, 10)]);

		plt.show(block=False)
		plt.savefig('./figures/confusion_matrix_{}.png'.format(name))

		print("[>] Saving confusion matrix as confusion_matrix_{}.png in ./figures folder in the directory".format(name))


# use CVXOPT
def main(verbose=True):
	# processing for training
	p = ProcessingForMulti(train_file="./dataset/train.csv", test_file="./dataset/test.csv")

	# apply model to it
	m_svm = MultiSVM(verbose, 'gaussian', 0.05, 1.0)
	m_svm.fit(p.data)

	predicted_labels = m_svm.predict(p.testdata)

	# draw confusion matrix for test data only
	m_svm.draw_confusion_matrix(p.testdata["label"], predicted_labels, name='multisvm_cvxopt_test_data')

	# train accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	print('[!] Computational time (of training): {}'.format(m_svm.time_taken))

	# test accuracy
	predicted_labels = m_svm.predict(p.data)
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on train set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))


# use LIBSVM
def main_use_libsvm(verbose=True, C=1.0, validation=False):
	p = ProcessingForMulti(train_file="./dataset/train.csv", test_file="./dataset/test.csv", validation=validation)

	s = MultiSVM_libsvm(verbose, C, kernel_type='gaussian')
	s.fit(p.data)

	predicted_labels = s.predict(p.testdata)

	# draw confusion matrix for test data only
	# s.draw_confusion_matrix(p.testdata["label"], predicted_labels, name='multisvm_libsvm_test_data')
	
	# test accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("\n[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	print('[!] Computational time (of training): {}'.format(s.time_taken))

	# # train accuracy
	# predicted_labels = s.predict(p.data)
	# accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.data["label"][i]])) / float(len(predicted_labels))
	# print("[*] Accuracy on train set: {0:.5f}".format(accuracy))
	# print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

	# validation set accuracy
	predicted_labels = s.predict(p.validationdata)
	accuracy2 = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.validationdata["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on validation set: {0:.5f}".format(accuracy2))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy2))

	return [accuracy, accuracy2]

# this is for variable values of c
def main_with_variable_c():

	c_values = [0.00001, 0.001, 1, 5, 10]
	accuracy_valid_set = []
	accuracy_test_set = []

	for c_value in c_values:

		print("[!] For C-value: {}".format(c_value))
		acc = main_use_libsvm(verbose=True, C=c_value, validation=True)

		accuracy_test_set.append(acc[0])
		accuracy_valid_set.append(acc[1])

	# with open('values.txt', 'w') as f:
	# 	f.write("Test Set")
	# 	f.write(str(accuracy_test_set) + "\n")
	# 	f.write("Validation Set")
	# 	f.write(str(accuracy_valid_set) + "\n")
	# 	f.write("C-Values")
	# 	f.write(str(c_values) + "\n")

	fig, ax = plt.subplots(figsize=(12, 12))

	ax.plot(c_values, accuracy_valid_set, label="Validation Data", color='blue')
	ax.plot(c_values, accuracy_test_set, label="Test Data", color='red')
	ax.legend(loc='upper right', fontsize='x-large')

	plt.xlabel('C (noise) - Values')
	plt.ylabel('Accuracies on datasets')
	plt.title('C-values vs Accuracies on Test and Validation Data')

	# plt.savefig
	# ax.set_xlim(left=0)
	ax.set_ylim(bottom=0)
	ax.set_xscale('log')


	if not os.path.exists("./figures"):
		os.makedirs("./figures")

	plt.savefig("./figures/validation_test_vs_c_values.png")

	print("[!] Data plotted and saved in the directory!")

if __name__ == '__main__':

	# main_use_libsvm(verbose=True)
	# main(verbose=True)
	main_with_variable_c()