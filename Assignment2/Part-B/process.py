"""Preprocessing for SVM"""

import csv
import pandas as pd
import numpy as np

class Processing:
	def __init__(self, train_file, test_file, d=1):
		self.d = d
		self.train_file = train_file
		self.test_file = test_file

		self.process_data()
		self.process_test_data()

	def process_data(self):
		"""Binary Classification"""

		label1 = self.d
		label2 = (self.d+1) % 10
		max_pixel = 255

		# print(label1, label2)
		data = {"data": [], "label":[]}
		df = pd.read_csv(self.train_file)

		i =0
		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != label1):
				continue

			data["data"].append([(x/max_pixel) for x in l[:-1]])
			data["label"].append(1)
			i+=1
			if i>100:
				break

		i = 0
		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != label2):
				continue

			data["data"].append([(x/max_pixel) for x in l[:-1]])
			data["label"].append(-1)
			i+=1
			if i>100:
				break

		# set data (train data)
		self.data = data
		# print(len(self.data["data"]))
		self.label1 = label1
		self.label2 = label2

	def process_test_data(self):
		"""Binary Classification"""

		label1 = self.d
		label2 = (self.d+1) % 10
		max_pixel = 255

		# print(label1, label2)
		testdata = {"data": [], "label":[]}
		df = pd.read_csv(self.test_file)

		i=0
		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != label1):
				continue

			testdata["data"].append([(x/max_pixel) for x in l[:-1]])
			testdata["label"].append(label1)
			i+=1
			if i>10:
				break

		i=0
		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != label2):
				continue

			testdata["data"].append([(x/max_pixel) for x in l[:-1]])
			testdata["label"].append(label2)
			i+=1
			if i>10:
				break

		# set data (train data)
		self.testdata = testdata
		# print(len(self.testdata["data"]))

	
	def train_and_test(self, ratio=0.9):
		num_examples = len(self.data["data"])

		train = {"data": [], "label": []}
		test = {"data": [], "label": []}

		partition = int(num_examples*ratio)

		# partition of dataset into train and test
		train["data"] = self.data["data"][:partition]
		train["label"] = self.data["label"][:partition]

		test["data"] = self.data["data"][partition:]
		test["label"] = self.data["label"][partition:]

		return {"train": train, "test": test}


class ProcessingForMulti:
	def __init__(self, train_file, test_file):
		self.train_file = train_file
		self.test_file = test_file

		self.process_data()
		self.process_test_data()

		print("[*] Processing data for multi-class classification!")

	def process_data(self):
		max_pixel = 255

		data = {"data": [], "label":[]}
		df = pd.read_csv(self.train_file)

		i = 0
		for index, rows in df.iterrows():
			l = list(rows)
			data["data"].append([(x/max_pixel) for x in l[:-1]])
			data["label"].append(l[784])
			i+=1
			if i>300:
				break

		data["data"] = np.array(data["data"])
		data["label"] =  np.array(data["label"])

		self.data = data

	def process_test_data(self):
		max_pixel = 255

		testdata = {"data": [], "label":[]}
		df = pd.read_csv(self.test_file)

		i = 0
		for index, rows in df.iterrows():
			l = list(rows)
			testdata["data"].append([(x/max_pixel) for x in l[:-1]])
			testdata["label"].append(l[784])
			i+=1
			if i>50:
				break

		testdata["data"] = np.array(testdata["data"])
		testdata["label"] =  np.array(testdata["label"])

		self.testdata = testdata


if __name__ == '__main__':
	p = Processing(train_file="./dataset/train.csv", test_file="./dataset/test.csv")
