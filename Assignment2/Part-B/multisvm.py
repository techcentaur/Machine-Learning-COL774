"""Multi-class classification SVM"""


import copy
import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)

from svm import SVM
from process import (Processing, ProcessingForMulti)

class MultiSVM:
	def __init__(self):
		self.gamma = 0.05
		self.noise = 1.0

		self.num_classes = 0
		self.binary_svm_instances = []

	def fit(self, data):
		unique_labels = np.unique(data["label"])
		self.num_classes = len(np.unique(data["label"]))

		# one-vs-one classifier method
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# SVM classifier has one pos and one neg id
				negative = data["label"] == unique_labels[i]
				positive = data["label"] == unique_labels[j]

				print(negative)
				print(positive)

				X_data = np.r_[data["data"][negative], data["data"][positive]]
				Y_data = np.r_[data["label"][negative], data["label"][positive]]

				# divide the data into two categories
				Y_data[Y_data == unique_labels[i]] = -1.0
				Y_data[Y_data == unique_labels[j]] = 1.0

				print("[*] Classifier: (i={}, j={})".format(i, j))

				print(X_data)

				svm_object = SVM(verbose=True)
				svm_object.fit({"data": X_data, "label": Y_data})

				self.binary_svm_instances.append(copy.deepcopy(svm_object))

	def predict(self, testdata):
		num_test_data = len(testdata["data"])

		print("[!] testing examples: {}".format(num_test_data))

		svm_instances = (self.num_classes * (self.num_classes-1))/2
		predictions = np.zeros((num_test_data, self.num_classes))

		svm_object_id = 0
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# get predicted label for each class and report the maximum wins
				pred = self.binary_svm_instances[svm_object_id].predict(testdata)

				# pred<0: label i
				predictions[pred<0, i] += 1.0
				# pred>0: label j
				predictions[pred>0, j] += 1.0

				svm_object_id += 1

		return np.argmax(predictions, axis=1)


def main():
	# processing for training
	p = ProcessingForMulti(train_file="./dataset/train.csv", test_file="./dataset/test.csv")

	# apply model to it
	m_svm = MultiSVM()
	m_svm.fit(p.data)
	predicted_labels = m_svm.predict(p.testdata)

	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))



if __name__ == '__main__':
	main()