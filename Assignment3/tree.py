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

	# print(df['X1'])
	for col in cont_var_list:
		median = df[col].median()
		df.loc[df[col].astype('float64') <= median, col] = 0
		df.loc[df[col].astype('float64') > median, col] = 1
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
		# represent = '[#]-------Node-------[#]\n'
		represent += "[*] Parent: \n{}\n".format(str(self.parent))
		represent += "[*] Label: {}\n".format(str(self.label))
		represent += "[*] Number of children: {}\n".format(str(len(self.children)))
		represent += "[>] Label Counts: {}\n".format(str(self.label_counts))
		represent += "[>] Splitting Feature: {}\n".format(str(self.i_am_splitting_by_feature))
		represent += "[>] Splitting Feature Value: {}\n".format(str(self.splitted_feature_value))

		return represent

	def have_children(self):
		return len(self.children) != 0

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

		labels_counter = Counter(list(of_list))
		_sum = 0
		for key in labels_counter:
			if labels_counter[key] != 0:
				x = float(labels_counter[key])/float(len(of_list))
				_sum -= ((x)*(log(x, 2)))
		# print(_sum, end=" ")
		return _sum

	def information_gain(self, X, Y, feat):
		"""get info gain when breaking at feat"""

		H = self.entropy(Y)
		m = len(X)
		for label in set(X[feat]):
			sub_labels = Y[X[feat]==label]
			H = H - ((float(len(sub_labels))/float(m))*(self.entropy(sub_labels)))

		return H

	def get_best_feature(self, X, Y, features_list):
		"""return best feature and gain by some heuristic: Information gain here"""

		best = {"feature": None, "gain": 0}
		for feat in features_list:
			gain = self.information_gain(X, Y, feat)
			# print(gain, end=" ")
			if gain > best["gain"]:
				best["gain"] = gain
				best["feature"] = feat

		return best

	@staticmethod
	def are_all_labels_same(Y):
		return len(set(Y)) <= 1

	def make_decision_tree(self, X, Y, root, features):
		try:
			self.base_node = self.__make_decision_tree__(X, Y, root, features)
		except Exception as e:
			print("[*] Build Tree Error!\n{}\n".format(e))
			return False
		return True


	def __make_decision_tree__(self, X, Y, root, features):
		if self.are_all_labels_same(Y):
			leaf_node = root
			leaf_node.label = Y.iloc[0] # if not work use list(set(Y))[0]
			leaf_node.label_counts = Counter(list(Y))
			return leaf_node

		if len(features) == 0:
			return self.best_label_available(Y, root)

		# print("len of feats: {}".format(len(features)))
		# print(features)
		# find the best feature with the max informational gain
		print("[>] get best feature")
		best = self.get_best_feature(X, Y, features)
		print(best)

		if best["gain"]==0 or best["feature"] is None:
			return self.best_label_available(Y, root)

		root.i_am_splitting_by_feature = best["feature"]
		root.label_counts = Counter(list(Y))
		# print(root)
		# print(set(X[best["feature"]]))
		# wait()
		for val in set(X[best["feature"]]):
			print(set(X[best["feature"]]))
			print(val)
			sub_part_X = X[X[best["feature"]] == val]
			sub_part_Y = Y[X[best["feature"]] == val]
			# print(sub_part_X)
			# print(sub_part_Y)
			self.count += 1
			print(self.count)

			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
				
			child.parent = root
			child.splitted_feature_value = val
			root.children.append(child)
			print(child)
			# print(child.parent)
			# break
			# print(list(features))
			features_remaining = list(features)
			features_remaining = features_remaining.remove(best["feature"])
			print(features_remaining)
			# self.__make_decision_tree__(X, Y, child, features_remaining)
			wait()


		return root

def _pred(node, X_1):
	# if no children - return label of node
	if not node.have_children:
		return node.label
	else:
		required_childs = []
		# get child which has splitted feature
		for child in node.children:
			if child.splitted_feature_value == X_1[node.i_am_splitting_by_feature]:
				required_childs.append(child)

		# if no child: return max label
		if(len(required_childs)==0):
			_label = max(node.labels_counter.iteritems(), key=operator.itemgetter(1))[0]
			return _label

		# else: recursively go in the child
		return _pred(required_childs[0], X_1)



def predict_accuracy(testX, testY, node):
	"""predict accuracy using node on testX with testY"""

	correctly_predicted = 0
	for example in range(len(testX)):
		if(_pred(node, testX.iloc[example])) == (testY.iloc[example]):
			correctly_predicted += 1

	return float(correctly_predicted)/len(testY)

def wait():
	while True:
		pass

def main(part):
	# data = pd.read_csv("/dataset/credit-cards.train.csv")
	# X_T = data.drop("default payment next month", 1)
	# Y_T = data["default payment next month"]

	data = pd.read_csv("./dataset/credit-cards.test.csv")
	X_t = (data.drop("Y", 1))
	X_t = X_t.iloc[1:, 1:]
	Y_t = (data["Y"])
	Y_t = Y_t.iloc[1:]
	assert str(type(Y_t)) == "<class 'pandas.core.series.Series'>"

	# print(Y_t)

	# data = pd.read_csv("/dataset/credit-cards.val.csv")
	# X_v = data.drop("default payment next month", 1)
	# Y_v = data["default payment next month"]

	if part.lower() =="a":
		print("[#] Part-A:")

		continous_values_to_boolean(X_t)
		print("[>] Working with {} columns\n".format(len(list(X_t.columns))))

		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_t, Y_t, Node(), list(X_t.columns))

		wait()
		print("[>] Decision Tree Built:\n")
		print("[.] Nodes: {}\n".format(dt.count))
		print("[.] Root-Node: {}\n".format(dt.base_node))

		accuracy = predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy: {}\n".format(accuracy))


if __name__ == '__main__':
	# args
	import sys
	part_name = sys.argv[1]
	# print(part_name)
	main(part_name)