"""
Hold my beer while I implement the neural network
"""

import pandas as pd 
import numpy as np
from scipy.special import expit

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

		for i in range(0, len(self.shape)-1):
			w = (2*(np.random.rand(self.shape[i+1], self.shape[i])) + 1)*0.01 #[-1,1]
			b = (2*(np.random.rand(self.shape[i+1], 1)) + 1)*0.01 #[-1,1]

			self.weights.append(w)
			self.biases.append(b)

	def __str__(self):
		print("[>] Neural Network:")
		print("\t[>] Shape: ", str(self.shape))
		print("\t[.] Learning Rate: ", self.learning_rate)
		print("\t[.] Batch Size: ", self.batch_size)
		print("\t[.] Max Epoch: ", self.max_epoch)
		return ""

	def sigmoid(self, x):
		"""1/(1+np.exp(-x))"""
		return expit(x)

	def sigmoid_derivative(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def sigmoid_der(self, a):
		return a*(1-a)

	def feed_forward(self, X):
		for i in range(len(self.weights)):
			z = self.weights[i] @ X
			z = z + np.tile(self.biases[i], z.shape[1])
			a = self.sigmoid(z)
			X = a
		return a

	def forward_pass(self, X, Y):
		activations = []
		for i in range(len(self.weights)):
			z = self.weights[i] @ X
			z = z + np.tile(self.biases[i], z.shape[1])
			a = self.sigmoid(z)
			activations.append(a)
			X = a
		return activations

	def calculate_deltas(self, X, Y, activations):
		deltas = []
		last_delta = -1*np.multiply((Y - activations[-1]), self.sigmoid_der(activations[-1]))
		deltas.append(last_delta)

		last_activation = len(activations)-2
		for i in range(len(self.weights)-1, 0, -1):
			delta = np.multiply((self.weights[i].T @ last_delta), self.sigmoid_der(activations[last_activation]))
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

	def fit(self, X, Y):
		max_batches = len(X) // self.batch_size
		bsz = self.batch_size
		epoch = 0
		while epoch < self.max_epoch:
			error = 0
			epoch += 1
			# print(max_batches)
			for i in range(max_batches):
				_X = X[i*bsz : bsz*(i+1)]
				_Y = Y[i*bsz : bsz*(i+1)]
				# print(_X)
				# print(_Y)
				# wait()
				loss = self.__train__(_X.T, _Y.T)
				error += loss
			# self.predict(X[:1])
			error = error/max_batches
			print("Epoch {e}| Error: {err}".format(e=epoch, err=error/self.batch_size))
			# input()

	def __train__(self, X, Y):
		activations = self.forward_pass(X, Y)
		# print(activations)
		deltas = self.calculate_deltas(X, Y, activations)
		# print(deltas)
		derivatives = self.get_derivatives(activations, deltas, X)
		# print(derivatives)
		# wait()
		self.update_weights_biases(derivatives, deltas)

		loss = self.get_loss(activations[-1]-Y)
		# print(loss)
		# wait()
		return loss

	def predict(self, X):
		Y = self.feed_forward(X.T)
		print(Y)

	def score(self, X, Y):
		Y_predict = self.feed_forward(X.T)
		same = (np.sum(np.argmax(Y_predict.T, axis=1) == np.argmax(Y, axis=1)))
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

if __name__ == '__main__':

	data = {
	"input_units": 85,
	"shape": [30],
	"output_units": 10,
	"learning_rate": 0.1,
	"batch_size": 100,
	"max_epoch": 100
	}

	nn = NeuralNetwork(data)
	X, Y = read_data("./dataset/one_hot_train.csv")
	# X = np.array([[1, 0, 1, 1, 0, 0],
	# 			[0, 0, 0, 0, 1, 1],
	# 			[0, 0, 1, 0, 1, 1],
	# 			[1, 1, 1, 0, 0, 0]])
	# Y = np.array([[0, 1],
	# 			[1, 0],
	# 			[1, 0],
	# 			[0, 1]
	# 			])

	print("[>] Input shape: ", X.shape)
	print("[>] Output shape: ", Y.shape)
	print(nn)

	nn.fit(X, Y)
	acc = nn.score(X, Y)
	print("[*] Accuracy {}".format(acc))
