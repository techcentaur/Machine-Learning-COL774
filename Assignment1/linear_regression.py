# Implementing Linear Regression: It takes time to do this too.
# Start: 3:34am
import numpy as np
import pandas as pd

class LinearRegression:
	def __init__(self):
		pass

	def fit(self, X, y):
		m = X.shape[0]
		n = X.shape[1]

		converged = False
		num_iterations = 0
		while not converged:
						




class ReadFiles:
	def __init__(self):
		pass

	def read_file(self, filenameX, filenameY):
		x = pd.read_csv(filenameX, 'r')
		x = x.as_matrix()
		# numpy function asmatrix() doesn't copy data look into that
		y = pd.read_csv(filenameY, 'r')
		y = y.as_matrix().flatten()
		
		return {"input": x, "target": y}

if __name__ == '__main__':
	r = ReadFiles()
	df = r.read_file("./data/linearX.csv", "./data/linearY.csv")
	print(df)
	# print(df["input"].shape)
	# print(df["target"].shape)