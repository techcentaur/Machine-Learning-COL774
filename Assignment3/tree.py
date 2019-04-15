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
import numpy as np

# my modules
from helper_tree import (get_nodes_stats_depth_wise, breadth_first_search, wait)

# find median and categorise continous variables

class ToBoolean:
	def __init__(self):
		"""Init: categorical attribute names"""

		self.cont_var_list = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
		self.medians_cats = {}

	def continous_data_by_calculating_medians(self, df):
		"""classify by calculating median"""

		for col in self.cont_var_list:
			if col in list(df.columns):
				median = df[col].median()

				self.medians_cats[col] = median
				df.loc[df[col].astype('float64') <= median, col] = 0
				df.loc[df[col].astype('float64') > median, col] = 1

	def continous_data_with_already_calculated_medians(self, df):
		"""classify by already calculated medians"""

		for col in self.cont_var_list:
			if col in list(df.columns):
				median = self.medians_cats[col]

				df.loc[df[col].astype('float64') <= median, col] = 0
				df.loc[df[col].astype('float64') > median, col] = 1


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

	def get_max_labels(self):
		label = max(self.label_counts.items(), key=operator.itemgetter(1))[0]
		return label

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
		if self.are_all_labels_same(Y):
			leaf_node = root
			leaf_node.label = list(set(Y))[0] # if not work use list(set(Y))[0]
			leaf_node.label_counts = Counter(list(Y))
			return leaf_node

		if len(features) == 0:
			return self.best_label_available(Y, root)

		best = self.get_best_feature(X, Y, features)

		if best["gain"] == 0:
			return self.best_label_available(Y, root)

		root.i_am_splitting_by_feature = best["feature"]
		root.label_counts = Counter(list(Y))

		for val in set(X[best["feature"]]):
			sub_part_X = X[X[best["feature"]] == val]
			sub_part_Y = Y[X[best["feature"]] == val]
			self.count += 1
			# print(self.count)
			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
				
			child.parent = root
			child.splitted_feature_value = val
			root.children.append(child)

			features_remaining = list(features)
			features_remaining.remove(best["feature"])

			self.__make_decision_tree__(sub_part_X, sub_part_Y, child, features_remaining)

		return root

	def _pred(self, node, X_1):
		# if no children - return label of node
		if not node.have_children():
			return node.get_max_labels()
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
			return self._pred(required_childs[0], X_1)


	def predict_accuracy(self, testX, testY, node):
		"""predict accuracy using node on testX with testY"""

		correctly_predicted = 0
		for example in range(len(testX)):
			if(self._pred(node, testX.iloc[example])) == (testY.iloc[example]):
				correctly_predicted += 1

		return float(correctly_predicted)/len(testY)


	def _pred_with_depth(self, node, X_1, depth):
		# if no children - return label of node
		if (not node.have_children()) or (depth == 0):
			return node.get_max_labels()
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
			return self._pred_with_depth(required_childs[0], X_1, depth-1)


	def predict_accuracy_with_depth(self, testX, testY, node, depth=0):
		"""predict accuracy using node on testX with testY"""

		correctly_predicted = 0
		for example in range(len(testX)):
			if(self._pred_with_depth(node, testX.iloc[example], depth)) == (testY.iloc[example]):
				correctly_predicted += 1

		return float(correctly_predicted)/len(testY)



class BuildTreeWhileP:
	"""Build Decision Tree with processing during
	And repeatition is allowed in splitting the node
	"""

	def __init__(self, count, cont):
		self.count = count
		self.base_node = None
		self.continous_data = cont
		self.report_numerical_data = {}
		for c in self.continous_data:
			self.report_numerical_data[c] = []

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
		return _sum

	def information_gain(self, X, Y, feat):
		"""get info gain when breaking at feat"""

		if feat in self.continous_data:
			H = self.entropy(Y)
			med = X[feat].median()

			part_Y1 = Y[X[feat].astype('float64') > med]
			entropy_Y1 = self.entropy(part_Y1)
			H = H - ((float(len(part_Y1))/float(len(X)))*(entropy_Y1))

			part_Y2 = Y[X[feat].astype('float64') <= med]
			entropy_Y2 = self.entropy(part_Y2)
			H = H - ((float(len(part_Y2))/float(len(X)))*(entropy_Y2))			

			return H
		else:
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
		if self.are_all_labels_same(Y):
			leaf_node = root
			leaf_node.label = Y.iloc[0] # if not work use list(set(Y))[0]
			leaf_node.label_counts = Counter(list(Y))
			return leaf_node

		if len(features) == 0:
			return self.best_label_available(Y, root)

		best = self.get_best_feature(X, Y, features)
		# print(best)

		if best["gain"] == 0:
			# print("[!] I break from here!\n")
			return self.best_label_available(Y, root)

		root.i_am_splitting_by_feature = best["feature"]
		root.label_counts = Counter(list(Y))

		if best["feature"] in self.continous_data:
			med = X[best["feature"]].median()

			self.report_numerical_data[best["feature"]].append(med)
			# ----break into 2 childs
			# LEFT
			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
			self.count += 1

			child.splitted_feature_value = med
			child.parent = root
			root.children.append(child)

			features_remaining = list(features)
			# features_remaining.remove(best["feature"])

			sub_part_X = X[X[best["feature"]].astype('float64') <= med]
			sub_part_Y = Y[X[best["feature"]].astype('float64') <= med]
			self.__make_decision_tree__(sub_part_X, sub_part_Y, child, features_remaining)

			# RIGHT
			try:
				child = Node()
			except Exception as e:
				print("[!] Couldn't create child node {}".format(str(e)))
			self.count += 1

			child.splitted_feature_value = med
			child.parent = root
			root.children.append(child)

			features_remaining = list(features)
			# features_remaining.remove(best["feature"])

			sub_part_X = X[X[best["feature"]].astype('float64') > med]
			sub_part_Y = Y[X[best["feature"]].astype('float64') > med]
			self.__make_decision_tree__(sub_part_X, sub_part_Y, child, features_remaining)

		else:
			for val in set(X[best["feature"]]):
				sub_part_X = X[X[best["feature"]] == val]
				sub_part_Y = Y[X[best["feature"]] == val]
				self.count += 1
				# print(self.count)
				try:
					child = Node()
				except Exception as e:
					print("[!] Couldn't create child node {}".format(str(e)))
					
				child.splitted_feature_value = val
				child.parent = root
				root.children.append(child)

				features_remaining = list(features)
				features_remaining.remove(best["feature"])

				self.__make_decision_tree__(sub_part_X, sub_part_Y, child, features_remaining)

		return root

	def _pred(self, node, X_1):
		# if no children - return label of node
		if not node.have_children():
			return node.get_max_labels()
		else:
			required_childs = []
			# get child which has splitted feature
			if node.i_am_splitting_by_feature in self.continous_data:
				med = node.children[0].splitted_feature_value
				try:
					if X_1[node.i_am_splitting_by_feature].astype(float64) <= med:
						required_childs.append(node.children[0])
					else:
						required_childs.append(node.children[1])
				except Exception as e:
					_label = max(node.label_counts.items(), key=operator.itemgetter(1))[0]
					return _label

			else:
				for child in node.children:
					if child.splitted_feature_value == X_1[node.i_am_splitting_by_feature]:
						required_childs.append(child)

			# if no child: return max label
			if(len(required_childs)==0):
				_label = max(node.label_counts.items(), key=operator.itemgetter(1))[0]
				return _label

			# else: recursively go in the child
			return self._pred(required_childs[0], X_1)


	def predict_accuracy(self, testX, testY, node):
		"""predict accuracy using node on testX with testY"""

		correctly_predicted = 0
		for example in range(len(testX)):
			if(self._pred(node, testX.iloc[example])) == (testY.iloc[example]):
				correctly_predicted += 1

		return float(correctly_predicted)/len(testY)


	def _pred_with_depth(self, node, X_1, depth):
		# if no children - return label of node
		if (not node.have_children()) or (depth == 0):
			return node.get_max_labels()
		else:
			required_childs = []
			# get child which has splitted feature
			if node.i_am_splitting_by_feature in self.continous_data:
				med = node.children[0].splitted_feature_value
				try:
					if X_1[node.i_am_splitting_by_feature].astype(float64) <= med:
						required_childs.append(node.children[0])
					else:
						required_childs.append(node.children[1])
				except Exception as e:
					_label = max(node.label_counts.items(), key=operator.itemgetter(1))[0]
					return _label

			else:
				for child in node.children:
					if child.splitted_feature_value == X_1[node.i_am_splitting_by_feature]:
						required_childs.append(child)

			# if no child: return max label
			if(len(required_childs)==0):
				_label = max(node.label_counts.items(), key=operator.itemgetter(1))[0]
				return _label

			# else: recursively go in the child
			return self._pred_with_depth(required_childs[0], X_1, depth-1)


	def predict_accuracy_with_depth(self, testX, testY, node, depth):
		"""predict accuracy using node on testX with testY"""

		correctly_predicted = 0
		for example in range(len(testX)):
			if(self._pred_with_depth(node, testX.iloc[example], depth)) == (testY.iloc[example]):
				correctly_predicted += 1

		return float(correctly_predicted)/len(testY)


def plot(dt, data, part_name, nodes_stats):
	"""hold my beer while I plot all these shitty data"""

	accuracies = {"node": [], "train": [], "test":[], "val":[]}

	for depth in nodes_stats:

		acc1 = dt.predict_accuracy_with_depth(data["X_train"], data["Y_train"], dt.base_node, depth)
		acc2 = dt.predict_accuracy_with_depth(data["X_test"], data["Y_test"], dt.base_node, depth)
		acc3 = dt.predict_accuracy_with_depth(data["X_val"], data["Y_val"], dt.base_node, depth)

		accuracies["node"].append(nodes_stats[depth])
		accuracies["train"].append(acc1)
		accuracies["test"].append(acc2)
		accuracies["val"].append(acc3)

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

	plt.savefig('plot_part_{}_.png'.format(str(part_name)))
	print("[*] Saved figure with name: plot_part_{}.png".format(str(part_name)))


def this_is_not_my_daddy_remove_it(node):
	if not node.parent is None:
		node.parent.children.remove(node)
		return node.parent
	return

def this_is_my_daddy(daddy, node):
	daddy.children.append(node)
	return True


def convert_into_one_hot_encoding(df):
	categorical_cols = ['X3','X4','X6','X7','X8','X9','X10','X11']

	# print(df.columns)
	for a in list(df.columns):
		tmp = df[a]
		del df[a]
		if a in categorical_cols:
			if a == 'X3':
				arr = tmp.values
				tttmp = []
				for i in arr:
					ttmp = [0]*7
					ttmp[int(i)] = 1
					tttmp.append(ttmp)
				tttmp = np.array(tttmp)
				for i in range(len(tttmp.T)):
					df[a+str(i)] = tttmp.T[i]
			elif a == 'X4':
				arr = tmp.values
				tttmp = []
				for i in arr:
					ttmp = [0]*4
					ttmp[int(i)] = 1
					tttmp.append(ttmp)
				tttmp = np.array(tttmp)
				for i in range(len(tttmp.T)):
					df[a+str(i)] = tttmp.T[i]
			else:
				arr = tmp.values
				tttmp = []
				for i in arr:
					ttmp = [0]*12
					ttmp[(int(i)+2)] = 1
					tttmp.append(ttmp)
				tttmp = np.array(tttmp)
				for i in range(len(tttmp.T)):
					df[a+str(i)] = tttmp.T[i]
		else:
			df[a] = tmp
	# print(df.columns)
	return df



def main(part, trainfp, testfp, valfp):
	data = pd.read_csv(trainfp)
	X_T = (data.drop("Y", 1))
	X_T = X_T.iloc[1:, 1:]
	Y_T = (data["Y"])
	Y_T = Y_T.iloc[1:]
	assert str(type(Y_T)) == "<class 'pandas.core.series.Series'>"

	data = pd.read_csv(testfp)
	X_t = (data.drop("Y", 1))
	X_t = X_t.iloc[1:, 1:]
	Y_t = (data["Y"])
	Y_t = Y_t.iloc[1:]
	assert str(type(Y_t)) == "<class 'pandas.core.series.Series'>"

	data = pd.read_csv(valfp)
	X_v = (data.drop("Y", 1))
	X_v = X_v.iloc[1:, 1:]
	Y_v = (data["Y"])
	Y_v = Y_v.iloc[1:]
	assert str(type(Y_v)) == "<class 'pandas.core.series.Series'>"


	start = time.time()
	if part==1:
		print("[#] Part-A:")

		tb_object = ToBoolean()
		tb_object.continous_data_by_calculating_medians(X_T)
		tb_object.continous_data_with_already_calculated_medians(X_t)
		tb_object.continous_data_with_already_calculated_medians(X_v)

		print("[>] Working with {} columns\n".format(len(list(X_T.columns))))

		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_T, Y_T, Node(), list(X_T.columns))

		print("\n[>] DECISION TREE BUILT:")
		print("[>] Nodes: {}".format(dt.count))
		print("[>] Root-Node: {}\n".format(dt.base_node))

		accuracy = dt.predict_accuracy(X_T, Y_T, dt.base_node)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = dt.predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = dt.predict_accuracy(X_v, Y_v, dt.base_node)
		print("[*] Accuracy Validation: {}".format(accuracy))

		print("[*] Time taken: {}\n".format(time.time()-start))

		nodes_stats = get_nodes_stats_depth_wise(dt.base_node)
		data = {
			"X_train": X_T,
			"Y_train": Y_T,
			"X_test": X_t,
			"Y_test": Y_t,
			"X_val": X_v,
			"Y_val": Y_v
		}

		plot(dt, data, part, nodes_stats)

	elif part==2:
		# hold my cosmo while I write the pruning part
		print("[#] Part-B: Post-Pruning part")

		tb_object = ToBoolean()
		tb_object.continous_data_by_calculating_medians(X_T)
		tb_object.continous_data_with_already_calculated_medians(X_t)
		tb_object.continous_data_with_already_calculated_medians(X_v)

		dt = BuildTree(count=1) # root at 1
		dt.make_decision_tree(X_T, Y_T, Node(), list(X_T.columns))

		print("\n[>] DECISION TREE BUILT:")
		print("[>] Nodes: {}".format(dt.count))

		tim1 = time.time()
		acc_val = dt.predict_accuracy(X_v, Y_v, dt.base_node)
		print("val time: ", time.time()-tim1)

		print("[*] Pruning Start!")
		nodes_deleted = 0
		nodes_list = breadth_first_search(dt.base_node)
		nodes_list.reverse()
		for node in nodes_list:
			daddy = this_is_not_my_daddy_remove_it(node)
			tmp_acc_val = dt.predict_accuracy(X_v, Y_v, dt.base_node)
			
			if tmp_acc_val >= acc_val:
				nodes_deleted += 1
				print("[-] New Accuracy: {}".format(tmp_acc_val))
				acc_val = tmp_acc_val
			else:
				# if nodes_deleted>=40:
				# 	break
				if daddy is not None:
					this_is_my_daddy(daddy, node)

		print("[*] Pruning Complete!")
		print("[!] Deleted nodes {}".format(nodes_deleted))

		accuracy = dt.predict_accuracy(X_T, Y_T, dt.base_node)
		print("[*] Accuracy Train: {i1}".format(i1=accuracy))
		accuracy = dt.predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy Test: {i1}".format(i1=accuracy))
		accuracy = dt.predict_accuracy(X_v, Y_v, dt.base_node)
		print("[*] Accuracy Validation: {i1}".format(i1=accuracy))

		print("[*] Time taken: {}\n".format(time.time()-start))

		nodes_stats = get_nodes_stats_depth_wise(dt.base_node)
		data = {
			"X_train": X_T,
			"Y_train": Y_T,
			"X_test": X_t,
			"Y_test": Y_t,
			"X_val": X_v,
			"Y_val": Y_v
		}

		plot(dt, data, part, nodes_stats)

	elif part==3:
		print("[#] Part-C | Pre-processing whilst building the tree")

		# continous numerical data columns
		continous_data = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

		dt = BuildTreeWhileP(count=1, cont=continous_data) # root at 1
		dt.make_decision_tree(X_T, Y_T, Node(), list(X_T.columns))

		print("\n[>] DECISION TREE BUILT:")
		print("[>] Nodes: {}".format(dt.count))
		print("[>] Root-Node: {}\n".format(dt.base_node))

		accuracy = dt.predict_accuracy(X_T, Y_T, dt.base_node)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = dt.predict_accuracy(X_t, Y_t, dt.base_node)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = dt.predict_accuracy(X_v, Y_v, dt.base_node)
		print("[*] Accuracy Validation: {}".format(accuracy))

		print("[*] Time taken: {}\n".format(time.time()-start))
		# wait()

		data = {
			"X_train": X_T,
			"Y_train": Y_T,
			"X_test": X_t,
			"Y_test": Y_t,
			"X_val": X_v,
			"Y_val": Y_v
		}

		# print(dt.report_numerical_data)
		nodes_stats = get_nodes_stats_depth_wise(dt.base_node)
		plot(dt, data, part_name, nodes_stats)

	elif part==4:
		# validation set accuracy vs
		# [min samples split, min samples leaf, max depth]
		print("[#] Part-D | Using DecisionTreeClassifier")

		from sklearn.tree import DecisionTreeClassifier

		# _max_depth=[2,5,10,20,100,1000]
		# _min_samples_leaf= [2,5,10,20,40,80,160,200,500,1000]
		# _min_samples_split= [2,5,10,20,40,80,160,200,500,1000,2000,4000,6000]
		# for i in _min_samples_leaf:
		min_samples_split = 1000
		min_samples_leaf = 10
		max_depth = 5

		print("[>] Min-samples-split: ", min_samples_split)
		print("[>] Min-samples-leaf: ", min_samples_leaf)
		print("[>] Max-depth: ", max_depth)

		clf = DecisionTreeClassifier(criterion="entropy",
									random_state=0,
									max_depth=max_depth,
									min_samples_split=min_samples_split,
									min_samples_leaf=min_samples_leaf
										)
		clf.fit(X_T, Y_T)

		accuracy = clf.score(X_T, Y_T)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = clf.score(X_t, Y_t)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = clf.score(X_v, Y_v)
		print("[*] Accuracy Validation: {}\n".format(accuracy))

		# print((clf.tree_).node_count)
		# -----------------------changing parameters

	elif part==5:
		from sklearn.tree import DecisionTreeClassifier
		print("[#] Part-E | Using One-hot encoding in categorical data")

		X_T = convert_into_one_hot_encoding(X_T)
		# wait()
		X_t = convert_into_one_hot_encoding(X_t)
		X_v = convert_into_one_hot_encoding(X_v)


		min_samples_split = 1000
		min_samples_leaf = 10
		max_depth = 5

		print("[>] Min-samples-split: ", min_samples_split)
		print("[>] Min-samples-leaf: ", min_samples_leaf)
		print("[>] Max-depth: ", max_depth)

		clf = DecisionTreeClassifier(criterion="entropy",
									random_state=0,
									max_depth=max_depth,
									min_samples_split=min_samples_split,
									min_samples_leaf=min_samples_leaf
										)
		clf.fit(X_T, Y_T)

		accuracy = clf.score(X_T, Y_T)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = clf.score(X_t, Y_t)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = clf.score(X_v, Y_v)
		print("[*] Accuracy Validation: {}\n".format(accuracy))

	elif part==6:
		from sklearn.ensemble.forest import RandomForestClassifier

		print("[#] Part-F | Random Forests Using One-hot encoding in categorical data")

		X_T = convert_into_one_hot_encoding(X_T)
		# wait()
		X_t = convert_into_one_hot_encoding(X_t)
		X_v = convert_into_one_hot_encoding(X_v)

		# _n_estimators = [10, 20, 50, 80, 100, 150, 200, 500]
		# _bootstrap = [True, False]
		# _max_features = ['auto', 'log2', None, 1, 4, 8, 10, 15, 20]


		# for i in _max_features:
		n_estimators = 40
		bootstrap = True
		max_features = 'log2'

		print("[>] N-estimators: ", n_estimators)
		print("[>] Bootstrap: ", bootstrap)
		print("[>] Max-features: ", max_features)


		# n_estimators = 10
		# bootstrap = True
		# max_features = 'auto'

		clf = RandomForestClassifier(criterion="entropy",
									max_features=max_features,
									n_estimators=n_estimators,
									bootstrap=bootstrap
										)
		clf.fit(X_T, Y_T)

		accuracy = clf.score(X_T, Y_T)
		print("[*] Accuracy Train: {}".format(accuracy))
		accuracy = clf.score(X_t, Y_t)
		print("[*] Accuracy Test: {}".format(accuracy))
		accuracy = clf.score(X_v, Y_v)
		print("[*] Accuracy Validation: {}\n".format(accuracy))


if __name__ == '__main__':
	# args
	import sys
	part_name = int(sys.argv[1])
	train_filepath = str(sys.argv[2])
	test_filepath = str(sys.argv[3])
	val_filepath = str(sys.argv[4])

	# print(part_name, train_filepath, test_filepath, val_filepath)
	main(part_name, train_filepath, test_filepath, val_filepath)


