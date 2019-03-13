import re
import os
import math

from random import randint

from preprocessing import (count_frequency, Preprocess)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt     

class NaiveBayes:
	def __init__(self, process, verbose=False):
		self.process = process

		self.vocabulary = set()
		self.class_words = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {} }
		
		self.labels = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
		self.class_priors = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

		self.verbose = verbose

	def fit(self, data):

		X = data["text"]
		Y = data["label"]
		m = len(X)

		if self.verbose:
			print("\n[>] Starting training with {} examples!".format(m))

		for label in Y:
			self.labels[str(label)] += 1
		for key in self.class_priors:
			self.class_priors[key] = (self.labels[key]/m) 

		# iteration over data space
		for x, y in zip(X, Y):
			freqs = count_frequency(x)

			for word, count in freqs.items():
				if word not in self.vocabulary:
					self.vocabulary.add(word)
				if word not in self.class_words[str(y)]:
					self.class_words[str(y)][word] = 0.0

				self.class_words[str(y)][word] += count

		if self.verbose:
			print("[#] Training complete!\n")


	def predict(self, data, text_normalised=False):
		test_data = data["text"]

		if self.verbose:
			print("\n[>] Starting the testing with {} examples!".format(len(test_data)))

		predicted_labels = []

		for x in test_data:
			counts = count_frequency(x)

			score = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
			log_scores = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

			for word, count in counts.items():
				# Add Laplace smoothing here
				for key in self.class_words:
					if word not in self.vocabulary:
						log_scores[key] = math.log((1)/(len(self.class_words[key]) + len(self.vocabulary)))
					else:
						log_scores[key] = math.log((self.class_words[key].get(word, 0.0) + 1)/(len(self.class_words[key]) + len(self.vocabulary)))
					
					score[key] += log_scores[key]
		
			# add class prior probs in log space
			for key in self.class_words:      
				score[key] += math.log(self.class_priors[key])

			prediction = max(score, key=score.get)
			predicted_labels.append(int(prediction))

		if self.verbose:
			print("[#] Testing complete!\n")

		return predicted_labels

	def predict_random(self, test_data, text_normalised=False):
		test_data = test_data["text"]

		if self.verbose:
			print("\n[>] Predicting Randomly with {} examples!".format(len(test_data)))

		predicted_labels = []

		for l in range(0, len(test_data)):
			i = randint(1, 5) #inclusive of 1 and 5
			predicted_labels.append(i)

		if self.verbose:
			print("[#] Testing complete!\n")

		return predicted_labels

	def predict_majority(self, test_data, text_normalised=False):
		test_data = test_data["text"]

		if self.verbose:
			print("\n[>] Predicting Majority with {} examples!".format(len(test_data)))

		predicted_labels = []

		max_label = max(self.labels, key=self.labels.get) 
		for l in range(0, len(test_data)):
			predicted_labels.append(int(max_label))

		if self.verbose:
			print("[#] Testing complete!\n")

		return predicted_labels

	def draw_confusion_matrix(self, actual_y, predicted_y, num=''):
		cm = confusion_matrix(actual_y, predicted_y)

		if not os.path.exists("./figures"):
			os.makedirs("./figures")

		fig, ax = plt.subplots(figsize=(10, 10))
		sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels');
		ax.set_ylabel('Actual labels');

		ax.set_title('Confusion Matrix | Part-A'); 
		ax.xaxis.set_ticklabels([str(i) for i in range(1, 6)]);
		ax.yaxis.set_ticklabels([str(i) for i in range(1, 6)]);


		plt.show(block=False)
		plt.savefig('./figures/confusion_matrix_{}.png'.format(num))

		if self.verbose:
			print("[>] Saving confusion matrix as confusion_matrix_{}.png".format(num))

	def f1_score(self, actual_y, predicted_y):
		sc = f1_score(actual_y, predicted_y, average=None)
		print("[.] Score per Class: {}".format(sc))

		sc = (f1_score(actual_y, predicted_y, average='macro'))
		print("[.] Macro Average Score: {}".format(sc))


def main(verbose):
	# create instance of Preprocess of training set
	process1 = Preprocess('./dataset/train.json', verbose, stem=False, stopwords=False, feature_technique="normal")
	process2 = Preprocess('./dataset/test.json', verbose, stem=False, stopwords=False, feature_technique="normal")
	
	model = NaiveBayes(process1, verbose)
	model.fit(process1.data)

	# predict on test data
	predicted_labels = model.predict(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))
	# print("[*] Test Error Rate: {}\n".format(1-accuracy))

	model.draw_confusion_matrix(process2.data["label"], predicted_labels, num='test_d_2')

	"""

	# predict on train data
	predicted_labels = model.predict(process1.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process1.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on train set: {0:.4f}".format(accuracy))
	# print("[*] Test Error Rate: {}\n".format(1-accuracy))
	

	# predicting randomly	
	predicted_labels = model.predict_random(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set (predicting radomly): {0:.4f}".format(accuracy))


	# predicting majority
	predicted_labels = model.predict_majority(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set (predicting majority): {0:.4f}".format(accuracy))


	# draw confusion matrix
	model.draw_confusion_matrix(process2.data["label"], predicted_labels, num='test_d')
	# model.f1_score(process2.data["label"], predicted_labels)	

	# draw confusion matrix
	# model.draw_confusion_matrix(process1.data["label"], predicted_labels, num='train_d')
	# model.f1_score(process1.data["label"], predicted_labels)	

	"""

def main1(verbose):
	# create instance of Preprocess of training set
	process1 = Preprocess('./dataset/train.json', verbose, stem=True, stopwords=True)
	process2 = Preprocess('./dataset/test.json', verbose, stem=True, stopwords=True)
	
	# fit model for naive bayes
	model = NaiveBayes(process1, verbose)
	model.fit(process1.data)

	# predict on test data
	predicted_labels = model.predict(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	model.f1_score(process2.data["label"], predicted_labels)	


	# with open('log.txt', 'w') as f:
	# 	f.write(str(model.class_words))
	# 	f.write(str(model.class_priors))
	# 	f.write(str(model.labels))
	# 	f.write(str(model.vocabulary))
	# 	# f.write(str(model.class_words))

	return accuracy

def main2(verbose):
	# create instance of Preprocess of training set
	process1 = Preprocess('./dataset/train.json', verbose, stem=True, stopwords=True, feature_technique="bigram")
	process2 = Preprocess('./dataset/test.json', verbose, stem=True, stopwords=True, feature_technique="bigram")
	
	# fit model for naive bayes
	model = NaiveBayes(process1, verbose)
	model.fit(process1.data)

	# predict on test data
	predicted_labels = model.predict(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	model.f1_score(process2.data["label"], predicted_labels)	

	return accuracy

def main3(verbose):
	# create instance of Preprocess of training set
	process1 = Preprocess('./dataset/train.json', verbose, stem=True, stopwords=True, feature_technique="advanced")
	process2 = Preprocess('./dataset/test.json', verbose, stem=True, stopwords=True, feature_technique="advanced")
	
	# fit model for naive bayes
	model = NaiveBayes(process1, verbose)
	model.fit(process1.data)

	# predict on test data
	predicted_labels = model.predict(process2.data, text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	model.f1_score(process2.data["label"], predicted_labels)	

	return accuracy

if __name__ == '__main__':

	main(True)

	# print("[!] Only stemming and stopwords")
	# a = main1(True)
	# print("[!] Only stemming and stopwords + Bigrams")
	# b = main2(True)
	# print("[!] Only stemming and stopwords + Adavanced detection!")
	# c = main3(True)

	# print("\n\n[whole dataset]\n\n")

	# m = max(a, b, c)
	# if m==a:
	# 	print("\nword\n")
	# 	process1 = Preprocess('./dataset/train_full.json', True, stem=True, stopwords=True, feature_technique="word")
	# 	process2 = Preprocess('./dataset/test.json', True, stem=True, stopwords=True, feature_technique="word")
		
	# 	# fit model for naive bayes
	# 	model = NaiveBayes(process1, True)
	# 	model.fit(process1.data)

	# 	# predict on test data
	# 	predicted_labels = model.predict(process2.data, text_normalised=True)

	# 	# get accuracy
	# 	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	# 	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	# 	model.f1_score(process2.data["label"], predicted_labels)	

	# elif m==b:
	# 	print("\nbigram\n")
	# 	process1 = Preprocess('./dataset/train_full.json', True, stem=True, stopwords=True, feature_technique="bigram")
	# 	process2 = Preprocess('./dataset/test.json', True, stem=True, stopwords=True, feature_technique="bigram")
		
	# 	# fit model for naive bayes
	# 	model = NaiveBayes(process1, True)
	# 	model.fit(process1.data)

	# 	# predict on test data
	# 	predicted_labels = model.predict(process2.data, text_normalised=True)

	# 	# get accuracy
	# 	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	# 	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	# 	model.f1_score(process2.data["label"], predicted_labels)	

	# elif m==c:
	# 	print("\nadvanced\n")
	# 	process1 = Preprocess('./dataset/train_full.json', True, stem=True, stopwords=True, feature_technique="advanced")
	# 	process2 = Preprocess('./dataset/test.json', True, stem=True, stopwords=True, feature_technique="advanced")
		
	# 	# fit model for naive bayes
	# 	model = NaiveBayes(process1, True)
	# 	model.fit(process1.data)

	# 	# predict on test data
	# 	predicted_labels = model.predict(process2.data, text_normalised=True)

	# 	# get accuracy
	# 	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == process2.data["label"][i]])) / float(len(predicted_labels))
	# 	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))

	# 	model.f1_score(process2.data["label"], predicted_labels)	

