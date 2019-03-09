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

		data = {"data": [], "label":[]}

		df = pd.read_csv(self.train_file)
		for index, rows in df.iterrows():
			l = list(rows)
			data["data"].append(l[1:])
			data["label"].append(l[0])

		self.data = data

if __name__ == '__main__':
	p = Processing(train_file="./dataset/train.csv")
	p.process_data()