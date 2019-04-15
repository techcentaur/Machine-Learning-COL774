"""
Hold my beer while I implement the neural network
"""

import time
import sys
import pandas as pd 
import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pprint import pprint

np.random.seed(1)

class NeuralNetwork:
	def __init__(self, *args):
		self.weights = []
		self.biases = []

		self.shape = [args[0]["input_units"]] + args[0]["shape"] + [args[0]["output_units"]]
		self.input_units = args[0]["input_units"]
		self.output_units = args[0]["output_units"]
		self.learning_rate = args[0]["learning_rate"]
		self.batch_size = args[0]["batch_size"]
		self.max_epoch = args[0]["max_epoch"]
		self.max_error_diff = args[0]["max_error_diff"]
		self.tolerance = args[0]["tolerance"]
		# sigmoid - false ------- relu - true
		self.activation_function = args[0]["activation"].lower()
		if self.activation_function == "sigmoid":
			self.activation_bool = False
		else:
			self.activation_bool = True

		for i in range(0, len(self.shape)-1):
			w = (2*(np.random.rand(self.shape[i+1], self.shape[i])) - 1) #[-1,1]
			b = (2*(np.random.rand(self.shape[i+1], 1)) - 1) #[-1,1]

			self.weights.append(w)
			self.biases.append(b)

	def __str__(self):
		print("[>] Neural Network:")
		print("\t[>] Shape: ", str(self.shape))
		print("\t[.] Learning Rate: ", self.learning_rate)
		print("\t[.] Batch Size: ", self.batch_size)
		print("\t[.] Max Epoch: ", self.max_epoch)
		print("\t[.] Tolerance: ", self.tolerance)
		print("\t[.] Activation Function for Hidden Layers: ", self.activation_function)
		return ""

	def sigmoid(self, x, act=False):
		"""
		1/(1+np.exp(-x)): Sigmoid
		max(0,z): relu
		"""
		if act:
			return np.where(x > 0, x, 0)
		else:
			return expit(x)

	def sigmoid_der(self, a, act=False):
		if act:
			return np.where(a>0, 1, 0)
		else:
			return a*(1-a)

	def feed_forward(self, X):
		for i in range(len(self.weights)):
			z = self.weights[i] @ X
			z = z + np.tile(self.biases[i], z.shape[1])
			if i==(len(self.weights)-1):
				a = self.sigmoid(z)
			else:
				a = self.sigmoid(z, self.activation_bool)
			X = a
		return a

	def forward_pass(self, X, Y):
		activations = []
		for i in range(len(self.weights)):
			z = self.weights[i] @ X
			z = z + np.tile(self.biases[i], z.shape[1])
			if i==(len(self.weights)-1):
				a = self.sigmoid(z)
			else:
				a = self.sigmoid(z, self.activation_bool)
			activations.append(a)
			X = a
		return activations

	def calculate_deltas(self, X, Y, activations):
		deltas = []
		last_delta = -1*np.multiply((Y - activations[-1]), self.sigmoid_der(activations[-1]))
		deltas.append(last_delta)

		last_activation = len(activations)-2
		for i in range(len(self.weights)-1, 0, -1):
			delta = np.multiply((self.weights[i].T @ last_delta), self.sigmoid_der(activations[last_activation], self.activation_bool))
			last_activation -= 1
			deltas.append(delta)
			last_delta = delta

		return deltas

	def get_derivatives(self, activations, deltas, X):
		j_derivatives = []
		activation_index = len(activations)-2
		delta_index = 0
		
		for i in range(len(activations)):
			if activation_index < 0:
				layer = X 
			else:
				layer = activations[activation_index] 
			derv = deltas[i] @ layer.T 
			j_derivatives.append(derv)
			activation_index -= 1
			delta_index += 1
			
		return j_derivatives

	def update_weights_biases(self, derivatives, deltas):
		derivatives.reverse()
		deltas.reverse()

		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - self.learning_rate*derivatives[i]
			self.biases[i] = self.biases[i] - self.learning_rate*(np.sum(deltas[i], axis=1).reshape(self.biases[i].shape[0], 1))

	def get_loss(self, delta):
		loss = (np.sum(np.sum(np.absolute(delta)**2, axis=1)/self.batch_size))
		return loss

	def fit(self, X, Y, tol=False):
		max_batches = len(X) // self.batch_size
		bsz = self.batch_size
		epoch = 0
		error_last = 0
		error_diff = 1
		while epoch < self.max_epoch and error_diff > self.max_error_diff:
			error = 0
			epoch += 1
			for i in range(max_batches):
				_X = X[i*bsz : bsz*(i+1)]
				_Y = Y[i*bsz : bsz*(i+1)]
				loss = self.__train__(_X.T, _Y.T)
				error += loss
			error = error/max_batches
			error_diff = abs(error-error_last)
			error_last=error
			
			if tol:
				if error_diff < self.tolerance:
					self.learning_rate = self.learning_rate/5

			print("Epoch {e}| Error: {err}".format(e=epoch, err=error))
			# input()

	def __train__(self, X, Y):
		activations = self.forward_pass(X, Y)
		deltas = self.calculate_deltas(X, Y, activations)
		derivatives = self.get_derivatives(activations, deltas, X)
		self.update_weights_biases(derivatives, deltas)

		loss = self.get_loss(activations[-1]-Y)
		return loss

	def predict(self, X):
		Y = self.feed_forward(X.T)
		print(Y)

	def score(self, X, Y, name=None):
		Y_predict = self.feed_forward(X.T)
		same = (np.sum(np.argmax(Y_predict.T, axis=1) == np.argmax(Y, axis=1)))

		if name is not None:
			draw_confusion_matrix(name, np.argmax(Y, axis=1), np.argmax(Y_predict.T, axis=1))
		return (float(same)/float(len(X)))

def wait():
	while True:
		pass	

def read_data(filepath):
	df = pd.read_csv(filepath, header=None)

	df = df.values
	dfX = df[:, :85]
	dfY = df[:, 85:]
	# wait()
	return dfX, dfY


def plot(name, base_list, list1, list2=None):
	if list2 is None:
		fig, ax = plt.subplots(figsize=(8, 8)) 
		fig.suptitle('Time taken vs Hidden Layers', fontsize=20)

		# plt.figure()
		plt.plot(base_list, list1, label='Time Taken to train', color='green')

		plt.legend(loc='best')
		plt.xlabel('[>] Units in Hidden Layer')
		plt.ylabel('[>] Time Taken to train')
		ax.set_ylim(bottom=0)

		plt.savefig(name)
		print("[*] Saved figure with name: {}".format(name))
	else:
		fig, ax = plt.subplots(figsize=(8, 8)) 
		fig.suptitle('Train and Test Accuracies vs Hidden Layers', fontsize=20)

		# plt.figure()
		plt.plot(base_list, list1, label='Train', color='red')
		plt.plot(base_list, list2, label='Test', color='blue')

		plt.legend(loc='best')
		plt.xlabel('[>] Units in Hidden Layer')
		plt.ylabel('[>] Accuracies')
		ax.set_ylim(bottom=0)

		plt.savefig(name)
		print("[*] Saved figure with name: {}".format(name))


def draw_confusion_matrix(name, actual_y, predicted_y):
	import seaborn as sns

	cm = confusion_matrix(actual_y, predicted_y)

	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

	# labels, title and ticks
	ax.set_xlabel('Predicted labels');
	ax.set_ylabel('Actual labels');

	ax.set_title('Confusion Matrix'); 
	ax.xaxis.set_ticklabels([str(i) for i in range(0, 10)]);
	ax.yaxis.set_ticklabels([str(i) for i in range(0, 10)]);

	plt.show(block=False)
	plt.savefig(name)

	print("[>] Saving confusion matrix as: {}".format(name))


if __name__ == '__main__':
	config_file_path = str(sys.argv[1])
	training_file_path = str(sys.argv[2])
	testing_file_path = str(sys.argv[3])

	with open(config_file_path) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	# print(content)
	input_units = int(content[0])
	output_units = int(content[1])
	batch_size = int(content[2])
	num_hidden = int(content[3])
	tmp_hidden = (content[4])
	tmp_hidden = tmp_hidden.split()
	hidden = []
	for idx in tmp_hidden:
		hidden.append(int(idx))
	activation = str(content[5])
	fixed_or_not = str(content[6])

	# wait()

	# hidden = [5, 10, 15, 20, 25]

	# time_list = []
	# train_acc = []
	# test_acc = []

	X, Y = read_data(training_file_path)
	X_test, Y_test = read_data(testing_file_path)

	# hidden = [25]

	# for i in range(len(hidden)):
	data = {
	"input_units": input_units,
	# "shape": [hidden[i], hidden[i]],
	"shape": hidden,
	"output_units": output_units,
	"learning_rate": 0.1,
	"batch_size": batch_size,
	"max_epoch": 1000,
	"max_error_diff": 1e-10,
	"tolerance": 1e-4,
	"activation": activation
	}
	pprint(data)
	# wait()

	nn = NeuralNetwork(data)
	print(nn)

	start_time = time.time()
	if fixed_or_not.lower() == "fixed":
		nn.fit(X, Y)
	else:
		nn.fit(X, Y, tol=True)

	end_time = time.time()
	time_taken = end_time - start_time
	print("[*] Time taken: {}".format(time_taken))

	acc = nn.score(X, Y)
	print("[*] Train Accuracy: {}".format(acc))
	acc1 = nn.score(X_test, Y_test, name="confusion_matrix_part_no_name_hidden_{n}.png".format(n=" "))
	print("[*] Test Accuracy: {}".format(acc1))

	# time_list.append(time_taken)
	# train_acc.append(acc)
	# test_acc.append(acc1)

		# in order: base, train, test
	# plot(name="plot_f2_train_test_vs_hidden.png", base_list=hidden, list1=train_acc, list2=test_acc)
	# plot(name="plot_f2_time_vs_hidden.png", base_list=hidden, list1=time_list, list2=None)
