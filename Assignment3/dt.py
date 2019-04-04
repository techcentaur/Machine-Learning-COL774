# this is for the first question related to Decision Trees:

# T = train, t = test, v = validation

import operator
import pandas as pd 
from math import log

from collections import Counter

# find median and categorise continous variables
def continous_values_to_boolean(df):
	"""classify by median"""

	cont_var_list = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

	for col in cont_var_list:
		median = df[col].median()
		df.loc[df[col] <= median, col] = 0
		df.loc[df[col] > median, col] = 1
		# changed in dataframe (no return)

class Node:
	"""One Node in Decision Tree"""
	def __init__(self):
		self.parent = None
		self.label = None
		self.children = []
		self.label_counts = None
		self.i_am_splitting_by_feature = None
		self.splitted_feature_value = None

	def __repr__(self):
		"""official unambigous representation"""

		represent = ''
		represent += "[*] Parent: {}\n".format(str(self.parent))
		represent += "[*] Label: {}\n".format(str(self.label))
		represent += "[*] Children: {}\n".format(str(self.children))
		represent += "[*] Label Counts: {}\n".format(str(self.label_counts))
		represent += "[>] Splitting Feature: {}\n".format(str(self.i_am_splitting_by_feature))
		represent += "[>] Splitting Feature Value: {}\n".format(str(self.splitted_feature_value))

		return represent


class BuildTree:
	"""Build Decision Tree"""
	def __init__(self, count):
		# kwargs.iteritems():
		self.count = count
		self.base_node = None

	@staticmethod
	def best_label_available(Y, node):
		"""fill the node with the best label in Y"""

		labels_counter = Counter(list(Y))
		label = max(labels_counter.iteritems(), key=operator.itemgetter(1))[0]
		
		node.label = label
		node.label_counts = labels_counter

		return node

	@staticmethod
	def entropy(of_list):
		"""get entropy of list"""

		labels_counter = Counter(list(of_list)):
		_sum = 0
		for key in labels_counter:
			if labels_counter[key] != 0:
				x = float(labels_counter[key])/float(len(of_list))
				_sum -= ((x)*(log(x, 2)))
		return _sum

	@staticmethod
	def information_gain(X, Y, feat):
		"""get info gain when breaking at feat"""

		H = entropy(Y)
		m = len(X)
		for label in set(X[feat]):
			sub_labels = Y[X[feat]==label]
			H = H - ((float(len(sub_labels))/float(m))*(entropy(sub_labels)))

		return H

	@staticmethod
	def get_best_feature(X, Y, features_list):
		"""return best feature and gain by some heuristic: Information gain here"""

		best = {"feature": None, "gain": 0}
		for feat in features_list:
			gain = information_gain(X, Y, feat)
			if gain > best["gain"]:
				best["gain"] = gain
				best["feature"] = feat

		return best

	def make_decision_tree(self, X, Y, root, features):
		self.base_node = self.__make_decision_tree__(X, Y, root, features)
		return True

	def __make_decision_tree__(self, X, Y, root, features):
		if are_all_labels_same(Y):
			leaf = root
			leaf.label = Y.iloc[0] # if not work use list(set(Y))[0]
			leaf.label_counts = Counter(list(Y))
			return leaf

		if len(features) == 0:
			return best_label_available(Y, root)

		# find the best feature with the max informational gain
		best = get_best_feature(X, Y, features)

		if best["gain"]==0 or best["feature"] is None:
			return best_label_available(Y, root)

		root.i_am_splitting_by_feature = best["feature"]
		root.labels_counter = Counter(list(Y))

		for val in set(X[best["feature"]]):
			sub_part_X = X[X[best["feature"]] == val]
			sub_part_Y = Y[X[best["feature"]] == val]
			self.count += 1

			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
				
			child.parent = root
			child.splitted_feature_value = value
			root.children.append(child)

			features_remaining = list(features).remove(best["feature"])
			__make_decision_tree__(self, X, Y, child, features_remaining)

		return root


def accuracy(node, testX, testY):
	"""predict accuracy using node on testX with testY"""

	for i in range(len(testX)):


def main(part):
	# data = pd.read_csv("/dataset/credit-cards.train.csv")
	# X_T = data.drop("default payment next month", 1)
	# Y_T = data["default payment next month"]

	data = pd.read_csv("/dataset/credit-cards.test.csv")
	X_t = data.drop("default payment next month", 1)
	Y_t = data["default payment next month"]

	# data = pd.read_csv("/dataset/credit-cards.val.csv")
	# X_v = data.drop("default payment next month", 1)
	# Y_v = data["default payment next month"]

	if part.lower() =="a":
		# part - a

		continous_values_to_boolean(X_t)
		print("[>] Working with {} columns\n".format(len(list(X_t.columns))))

		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_t, Y_t, Node(), list(X_t.columns))

		print("[*] Decision Tree Built:\n")
		print("[.] Nodes: {}\n".format(dt.count))
		print("[.] Root-Node: {}\n".format(repr(dt.root)))

