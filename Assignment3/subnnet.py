# Hold my beer while I write the neural network algorithm

# my modules
import _net
# library modules
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


np.random.seed(1)

class NeuralNetwork:
	def __init__(self, input_units, shape, outputs_units, batch_size, learning_rate):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.input_units = input_units
		self.outputs_units = outputs_units
		# shape shall contain all units (design choice)
		self.shape = [input_units] + shape + [outputs_units]

		self.weights = self.init_weights(self.shape)

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def sigmoid_derivative(x):
		return NeuralNetwork.sigmoid(x) * NeuralNetwork.sigmoid(1 - x)

	def init_weights(self, network_shape):
		weight_arrays = []
		for i in range(0, len(network_shape) - 1):
			cur_idx = i
			next_idx = i + 1

			weight_array = 2*np.random.rand(network_shape[next_idx], network_shape[cur_idx]) - 1
			weight_arrays.append(weight_array)

		return weight_arrays


	def predict(self, input1):
		current_input = input1
		outputs = []
		for network_weight in self.weights:
			current_output_temp = np.dot(network_weight, current_input)

			current_output = self.sigmoid(current_output_temp)
			outputs.append(current_output)
			current_input = current_output

		return current_output.T

	def score(self, testX, testY):
		correct_prediction = 0

		testX = testX.T 
		testY = testY.T

		for i in range(len(testX)):
			tmp = self.predict(testX[i].T)
			tmp = np.where(tmp > 0.5, 1, 0)
			if np.all(tmp == testY[i]):
				correct_prediction += 1
				# print(correct_prediction)
				# wait()
		print(correct_prediction)
		accuracy = float(correct_prediction)/float(len(testX))
		return accuracy


	def fit(self, X, Y):
		# X: (85, 25010) | Y: (10, 25010)
		max_batchs = len(X.T)//self.batch_size
		b = self.batch_size

		# X = X.T
		# Y = Y.T
		# print(_X.shape)
		# for i in range(max_batchs-1):
		# # print(_Y.shape)
		# wait()
		# X, Y = shuffle(X.T, Y.T)
		_iter = 0
		weight_arrays = self.weights
			# _X = X.T[i*b : (i+1)*b]
			# _Y = Y.T[i*b : (i+1)*b]
			# # print(_X)
			# print(_Y)
			# continue
		while True:
			error = 0
			for i in range(len(X.T)):
				_X = (X.T[0]).reshape(1,-1)
				_Y = (Y.T[0]).reshape(1,-1)
				print(_X.shape)
				print(_Y.shape)
				print(_X)
				print(_Y)
				wait()
				_iter += 1
				# takes input as X:(features, number_of_examples)
				weight_arrays, loss = self.__train_network__(_X.T, _Y.T, self.learning_rate, self.shape, weight_arrays)

				error += loss
				# print(loss)
			print(error)
			# if(loss < 0.0001):
			# 	print("[>] Coverged at iteration: {i} | Loss {l}".format(i=_iter, l=loss))
			# 	break

		self.weights = weight_arrays


	def __train_network__(self, input1, output, learning_rate, network_shape, network_weights):
		current_input = input1

		outputs = []
		for network_weight in network_weights:
			current_output_temp = np.dot(network_weight, current_input)

			current_output = self.sigmoid(current_output_temp)
			outputs.append(current_output)
			current_input = current_output

		deltas = []

		final_error = output - outputs[len(outputs)-1]
		final_delta = final_error
		deltas.append(final_delta)

		# print(final_delta)
		cur_delta = final_delta
		back_idx = len(outputs) - 2

		for network_weight in network_weights[::-1][:-1]:
			next_error = np.dot(network_weight.T, cur_delta)
			next_delta = next_error * self.sigmoid_derivative(outputs[back_idx])
			deltas.append(next_delta)
			cur_delta = next_delta
			back_idx -= 1

		cur_weight_idx = len(network_weights) - 1

		for delta in deltas:
			input_used = None
			if cur_weight_idx - 1 < 0:
				input_used = input1
			else:
				input_used = outputs[cur_weight_idx - 1]

			network_weights[cur_weight_idx] += learning_rate*np.dot(delta, input_used.T)
			cur_weight_idx -= 1


		loss = np.sum((np.sum(np.absolute(final_delta), axis=0)), axis=0)/(2*len(input1.T))
		# print(loss)
		return network_weights, loss

def wait():
	while True:
		pass	

def read_data(filepath):
	df = pd.read_csv(filepath, header=None)

	df = df.values
	dfX = df[:, :85]
	dfY = df[:, 85:]
	return dfX, dfY

if __name__ == '__main__':

	X, Y = read_data("./dataset/one_hot_train.csv")
	X = X.T
	Y = Y.T
	print(X.shape)
	print(Y.shape)
	# wait()
	# print(X.T[0])

	inputs = np.array([[0, 0, 1, 1],
				   [1, 1, 1, 1], 
				   [1, 0, 1, 1]]).T

	outputs = np.array([[0],
						[1],
						[1]]).T

	input_units = 85
	shape = [10]
	outputs_units = 10
	batch_size = 100
	learning_rate = 0.5

	# wait()
	nn = NeuralNetwork(input_units, shape, outputs_units, batch_size, learning_rate)
	nn.fit(X, Y)
	# wait()

	# test_input = np.array([[1, 0, 1, 1], [0, 0, 0, 1]]).T

	acc = nn.score(X, Y)
	print("[>] Accuracy: {}%".format(acc*100))
