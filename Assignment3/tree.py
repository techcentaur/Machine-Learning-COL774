# this is for the first question related to Decision Trees:

# T = train, t = test, v = validation
import time
import operator
import pandas as pd 
from math import log

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# find median and categorise continous variables
def continous_values_to_boolean(df):
	"""classify by median"""

	cont_var_list = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
	# cont_var_list = ['X1', 'X5']

	# print(df['X1'])
	for col in cont_var_list:
		if col in list(df.columns):
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
		# represent += "[*] Parent: \n{}\n".format(str(self.parent))
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
		# kwargs.items():
		self.count = count
		self.base_node = None

	@staticmethod
	def best_label_available(Y, node):
		"""fill the node with the best label in Y"""

		labels_counter = Counter(list(Y))
		label = max(labels_counter.items(), key=operator.itemgetter(1))[0]
		
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
		self.base_node = self.__make_decision_tree__(X, Y, root, features)


	def __make_decision_tree__(self, X, Y, root, features):
		# print(features)
		if self.are_all_labels_same(Y):
			# print("Yes, all labels are same!\n")
			leaf_node = root
			leaf_node.label = Y.iloc[0] # if not work use list(set(Y))[0]
			leaf_node.label_counts = Counter(list(Y))
			return leaf_node

		if len(features) == 0:
			# print("empty")
			return self.best_label_available(Y, root)

		# print("len of feats: {}".format(len(features)))
		# print(X)
		# print(Y)
		# find the best feature with the max informational gain
		# print("[>] get best feature")
		best = self.get_best_feature(X, Y, features)
		# print(best)

		if best["gain"] == 0:
			# print("[!] I break from here!\n")
			return self.best_label_available(Y, root)

		root.i_am_splitting_by_feature = best["feature"]
		root.label_counts = Counter(list(Y))
		# print(root)
		# print(set(X[best["feature"]]))
		# wait()
		for val in set(X[best["feature"]]):
			# print(set(X[best["feature"]]))
			# print("Value", val)
			sub_part_X = X[X[best["feature"]] == val]
			sub_part_Y = Y[X[best["feature"]] == val]
			# print(sub_part_X)
			# print(sub_part_Y)
			self.count += 1
			# print(self.count)
			# wait()
			# continue

			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
				
			child.parent = root
			child.splitted_feature_value = val
			root.children.append(child)
			# print(child)
			# wait()
			# continue
			# print(child.parent)
			# break
			# print(list(features))
			features_remaining = list(features)
			features_remaining.remove(best["feature"])
			# print(features_remaining)
			self.__make_decision_tree__(sub_part_X, sub_part_Y, child, features_remaining)

		# wait()
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
			_label = max(node.label_counts.items(), key=operator.itemgetter(1))[0]
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


# Hold my beer while I write all these helper functions
def breadth_first_search(node):
	nodes_list = []
	tmplist = []
	tmplist.append(node)

	while len(tmplist) != 0:
		tmp = tmplist.pop(0)
		nodes_list.append(tmp)
		for child in tmp.children:
			tmplist.append(child)

	# print(nodes_list)
	nodes_list.reverse() #from leafs to root
	return nodes_list


def plot(base_node, data):
	nodes_list = breadth_first_search(base_node)
	print("[*] BFS done! Got list of nodes\n")
	# indexed_node_list = [x for x in range(1, len(nodes_list)+1)]

	print("[.] Plotting {} points!\n".format(int(len(nodes_list)/500)))
	# random.shuffle(indexed_node_list)
	
	accuracies = {"node": [], "train": [], "test":[], "val":[]}
	for index in range(1, len(nodes_list), 500):
		accuracies["train"].append(predict_accuracy(data["X_train"], data["Y_train"], nodes_list[index]))
		accuracies["test"].append(predict_accuracy(data["X_test"], data["Y_test"], nodes_list[index]))
		accuracies["val"].append(predict_accuracy(data["X_val"], data["Y_val"], nodes_list[index]))
		accuracies["node"].append(index)

	for key in accuracies:
		accuracies[key].reverse()
	
	print("[.] Accuracies per node calculated!\n")
	# plot my horses
	fig, ax = plt.subplots(figsize=(12, 12)) 
	# plt.figure()
	plt.plot(accuracies["node"], accuracies["train"], label='Train', color='red')
	plt.plot(accuracies["node"], accuracies["test"], label='Test', color='blue')
	plt.plot(accuracies["node"], accuracies["val"], label='Validation', color='green')

	plt.legend(loc='best')
	plt.xlabel('Number of nodes')
	plt.ylabel('Accuracies')
	ax.set_ylim(bottom=0)

	plt.savefig('plot_part_a.png')

def this_is_not_my_daddy_remove_it(node):
	if node.parent is None:
		return
	else:
		node.parent.children.remove(node)
		return node.parent

def this_is_my_daddy(daddy, node):
	daddy.children.append(node)
	return True

def main(part):
	data = pd.read_csv("./dataset/credit-cards.train.csv")
	X_T = (data.drop("Y", 1))
	X_T = X_T.iloc[1:50, 1:]
	Y_T = (data["Y"])
	Y_T = Y_T.iloc[1:50]
	assert str(type(Y_T)) == "<class 'pandas.core.series.Series'>"

	data = pd.read_csv("./dataset/credit-cards.test.csv")
	X_t = (data.drop("Y", 1))
	X_t = X_t.iloc[1:50, 1:]
	Y_t = (data["Y"])
	Y_t = Y_t.iloc[1:50]
	assert str(type(Y_t)) == "<class 'pandas.core.series.Series'>"

	data = pd.read_csv("./dataset/credit-cards.val.csv")
	X_v = (data.drop("Y", 1))
	X_v = X_v.iloc[1:50, 1:]
	Y_v = (data["Y"])
	Y_v = Y_v.iloc[1:50]
	assert str(type(Y_v)) == "<class 'pandas.core.series.Series'>"


	start = time.time()
	if part.lower() =="a":
		print("[#] Part-A:")

		continous_values_to_boolean(X_T)

		print("[>] Working with {} columns\n".format(len(list(X_T.columns))))

		# print(X_t)
		# wait()
		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_T, Y_T, Node(), list(X_T.columns))

		print("\n[>] DECISION TREE BUILT:")
		print("[>] Nodes: {}".format(dt.count))
		print("[>] Root-Node: {}\n".format(dt.base_node))

		accuracy = predict_accuracy(X_T, Y_T, dt.base_node)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = predict_accuracy(X_v, Y_v, dt.base_node)
		print("[*] Accuracy Validation: {}".format(accuracy))

		print("[*] Time taken: {}\n".format(time.time()-start))

		data = {
			"X_train": X_T,
			"Y_train": Y_T,
			"X_test": X_t,
			"Y_test": Y_t,
			"X_val": X_v,
			"Y_val": Y_v
		}

		# plot(dt.base_node, data)

	elif part.lower() =="b":
		# hold my cosmo while I write the pruning part
		print("[#] Part-B: Post-Pruning part")

		continous_values_to_boolean(X_T)

		print("[>] Working with {} columns\n".format(len(list(X_T.columns))))

		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_T, Y_T, Node(), list(X_T.columns))

		print("\n[>] DECISION TREE BUILT:")
		print("[>] Nodes: {}".format(dt.count))

		oldaccuracy1 = predict_accuracy(X_T, Y_T, dt.base_node)
		oldaccuracy2 = predict_accuracy(X_t, Y_t, dt.base_node)
		oldaccuracy3 = predict_accuracy(X_v, Y_v, dt.base_node)

		acc_val = predict_accuracy(X_v, Y_v, dt.base_node)

		print("[*] Pruning Start!")
		nodes_deleted = 0
		nodes_list = breadth_first_search(dt.base_node)
		
		for node in nodes_list:
			daddy = this_is_not_my_daddy_remove_it(node)
			tmp_acc_val = predict_accuracy(X_v, Y_v, dt.base_node)
			
			if tmp_acc_val > acc_val:
				nodes_deleted += 1
				# print("[-] Accuracy difference: {}".format(tmp_acc_val - acc_val))
				acc_val = tmp_acc_val
			else:
				if daddy is not None:
					this_is_my_daddy(daddy, node)

		print("[*] Pruning Complete!")
		print("[!] Deleted nodes {}".format(nodes_deleted))

		accuracy = predict_accuracy(X_T, Y_T, dt.base_node)
		print("[*] Accuracy Train: {i1} | Diff after pruning: {i2}".format(i1=round(accuracy, 5), i2=round(accuracy-oldaccuracy1, 5)))
		accuracy = predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy Test: {i1} | Diff after pruning: {i2}".format(i1=round(accuracy, 5), i2=round(accuracy-oldaccuracy2, 5)))
		accuracy = predict_accuracy(X_v, Y_v, dt.base_node)
		print("[*] Accuracy Validation: {i1} | Diff after pruning: {i2}".format(i1=round(accuracy, 5), i2=round(accuracy-oldaccuracy3, 5)))

		print("[*] Time taken: {}\n".format(time.time()-start))

		data = {
			"X_train": X_T,
			"Y_train": Y_T,
			"X_test": X_t,
			"Y_test": Y_t,
			"X_val": X_v,
			"Y_val": Y_v
		}
		# plot(dt.base_node, data)



if __name__ == '__main__':
	# args
	import sys
	part_name = sys.argv[1]
	# print(part_name)
	main(part_name)