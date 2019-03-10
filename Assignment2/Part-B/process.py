"""Preprocessing for SVM"""

import csv
import pandas as pd
class Processing:
	def __init__(self, train_file, d=1):
		self.d = d
		self.train_file = train_file

	def process_data(self):
		"""Binary Classification"""

		label1 = self.d
		label2 = (self.d+1) % 10
		max_pixel = 255

		# print(label1, label2)
		data = {"data": [], "label":[]}
		df = pd.read_csv(self.train_file)

		i = 0
		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != 1):
				continue

			data["data"].append([(x/max_pixel) for x in l[:-1]])
			data["label"].append(l[784])

		for index, rows in df.iterrows():
			l = list(rows)

			if (l[784] != 2):
				continue

			data["data"].append([(x/max_pixel) for x in l[:-1]])
			data["label"].append(-1)

			# if i>500:
			# 	break
			# i+=1
			# # print(data)
			# # break

		self.data = data
		# print(data)

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


if __name__ == '__main__':
	p = Processing(train_file="./dataset/train.csv")
	p.process_data()