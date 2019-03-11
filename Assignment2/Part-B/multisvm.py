"""Multi-class classification SVM"""

import os
import copy
import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from cvxopt import (matrix, solvers)
from sklearn.metrics import confusion_matrix
import seaborn as sns

from svm import SVM
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

		# one-vs-one classifier method
		for i in range(self.num_classes):
			for j in range(i+1, self.num_classes):

				# SVM classifier has one pos and one neg id
				negative = data["label"] == unique_labels[i]
				positive = data["label"] == unique_labels[j]

				# positive = positive.astype(int)
				# negative = negative.astype(int)

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

	def predict(self, testdata):
		num_test_data = len(testdata["data"])

		if self.verbose:
			print("[!] testing examples: {}".format(num_test_data))

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

		fig, ax = plt.subplots()
		sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels');
		ax.set_ylabel('Actual labels');

		ax.set_title('Confusion Matrix | Part-B | Multi-Class'); 
		ax.xaxis.set_ticklabels([str(i) for i in range(0, 10)]);
		ax.yaxis.set_ticklabels([str(i) for i in range(0, 10)]);

		plt.show(block=False)
		plt.savefig('./figures/confusion_matrix_{}.png'.format(name))

		if self.verbose:
			print("[>] Saving confusion matrix as confusion_matrix_{}.png".format(name))


def main():
	# processing for training
	p = ProcessingForMulti(train_file="./dataset/train.csv", test_file="./dataset/test.csv")

	# apply model to it
	m_svm = MultiSVM(False, 'gaussian', 0.05, 1.0)
	m_svm.fit(p.data)

	predicted_labels = m_svm.predict(p.testdata)
	# draw confusion matrix
	m_svm.draw_confusion_matrix(p.testdata["label"], predicted_labels)

	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == p.testdata["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.5f}".format(accuracy))
	print("[*] Test Error Rate: {0:.5f}".format(1-accuracy))

if __name__ == '__main__':
	main()