import re
import math

from random import randint

from preprocessing import (count_frequency, Preprocess)
from sklearn.metrics import confusion_matrix

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
			if not text_normalised:
				x = self.process.normalise_data(x)

			counts = count_frequency(x)

			score = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
			log_scores = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

			for word, count in counts.items():
				if word not in self.vocabulary:
					continue

				# Add Laplace smoothing here
				for key in self.class_words:
					# sum 1 in numerator; 5(no of cats) in denominator
					log_scores[key] = math.log((self.class_words[key].get(word, 0.0) + 1)/(len(self.class_words[key]) + 5))
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

		max_label = max(self.class_priors, key=self.class_priors.get) 
		for l in range(0, len(test_data)):
			predicted_labels.append(int(max_label))

		if self.verbose:
			print("[#] Testing complete!\n")

		return predicted_labels

	def draw_confusion_matrix(self, actual_y, predicted_y, num=1):
		cm = confusion_matrix(actual_y, predicted_y)

		fig, ax = plt.subplots()
		sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels');
		ax.set_ylabel('Actual labels');

		ax.set_title('Confusion Matrix | Part-A'); 
		ax.xaxis.set_ticklabels([str(i) for i in range(1, 6)]);
		ax.yaxis.set_ticklabels([str(i) for i in range(1, 6)]);

		plt.show(block=False)
		plt.savefig('./confusion_matrix_{}.png'.format(str(num)))

		if self.verbose:
			print("[>] Saving confusion matrix as confusion_matrix_{}.png".format(str(num)))


def main(verbose):
	# create instance of Preprocess of training set
	process1 = Preprocess('./dataset/subset.json', verbose, stem=True, stopwords=True)
	
	# divide the data
	data = process1.train_and_test()
	
	# fit model for naive bayes
	model = NaiveBayes(process1, verbose)
	model.fit(data["train"])

	# predict on data (text will normalise itself)
	predicted_labels = model.predict(data["test"], text_normalised=True)

	# get accuracy
	accuracy = float(sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == data["test"]["label"][i]])) / float(len(predicted_labels))
	print("[*] Accuracy on test set: {0:.4f}".format(accuracy))
	
	# draw confusion matrix
	model.draw_confusion_matrix(data["test"]["label"], predicted_labels, num=3)
	

if __name__ == '__main__':
	main(False)