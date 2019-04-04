# this is for the first question related to Decision Trees:

# T = train, t = test, v = validation

import operator
import pandas as pd 
from math import log

from collections import Counter
class DecisionTrees:
	def __init__(self):
		pass

# read data
# data = pd.read_csv("/dataset/credit-cards.train.csv")
# X_T = data.drop("default payment next month", 1)
# Y_T = data["default payment next month"]

data = pd.read_csv("/dataset/credit-cards.test.csv")
X_t = data.drop("default payment next month", 1)
Y_t = data["default payment next month"]

# data = pd.read_csv("/dataset/credit-cards.val.csv")
# X_v = data.drop("default payment next month", 1)
# Y_v = data["default payment next month"]

# find median and categorise continous variables

def continous_values_to_boolean(df):
	"""classify by median"""

	cont_var_list = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

	for col in col_list:
		median = df[col].median()
		df.loc[df[col] <= median, col] = 0
		df.loc[df[col] > median, col] = 1
		# changed in dataframe (no return)

class Tree:
	def __init__(self):
		self.children = []

		self.label = None
		self.parent = None
		self.label_counts = None

	def __repr__(self):
		"""official unambigous representation"""

		represent = ''
		represent += "[*] Parent: {}\n".format(str(self.parent))
		represent += "[*] Label: {}\n".format(str(self.label))
		represent += "[*] Children: {}\n".format(str(self.children))
		represent += "[*] Label Counts: {}\n".format(str(self.label_counts))


		return represent


class BuildTree:
	def __init__(self, count):
		# kwargs.iteritems():
		self.count = count

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
		labels_counter = Counter(list(of_list)):
		_sum = 0
		for key in labels_counter:
			if labels_counter[key] != 0:
				x = float(labels_counter[key])/float(len(of_list))
				_sum -= ((x)*(log(x, 2)))
		return _sum

	@staticmethod
	def information_gain(X, Y, feat):
		


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


	@staticmethod
	def make_decision_tree(X, Y, root, features):
		
		# are all the labels same?
		if are_all_labels_same(Y):
			leaf = root
			leaf.label = Y.iloc[0] # if not work use list(set(Y))[0]
			leaf.label_counts = Counter(list(Y))
			return leaf

		# is features-list empty?
		if len(features) == 0:
			return best_label_available(Y, root)


		# find the best feature with the max informational gain
		get_best_feature(X, Y, features)











continous_values_to_boolean(X_t)
print("[>] Working with {} columns\n".format(len(list(X_t.columns))))

dt = BuildTree(count=1) # root at 1
dt.make_decision_tree(X_t, Y_t, Tree(), list(X_t))
