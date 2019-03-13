import re
import ast
import json
import click
import pprint

import nltk
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from utils import json_reader, getStemmedDocuments

class Preprocess:
	def __init__(self):
		iter_over_data = json_reader(file_path)
		self.data = {"text": [], "label": []}
		while True:
			try:
				raw_datum = next(iter_over_data)
			except StopIteration:
				break

			self.data["text"].append(self.normalise_data(raw_datum["text"]))
			self.data["label"].append(int(raw_datum["stars"]))
		
	def normalise_data(self, text):
		text = getStemmedDocuments(text)
		return text
		# bigrms = list(nltk.bigrams(text))

def count_frequency(tokens):	
	count = {}
	for t in tokens:
		count[t] = count.get(t, 0.0) + 1.0
	return count


if __name__ == '__main__':
	processing = Preprocess(file_path="./dataset/sample.json", verbose=True, stem=True, stopwords=True, feature_technique="word")
	print(processing.data)





class NaiveBayes:
	def __init__(self, process, verbose=False):
		self.vocabulary = set()
		self.class_words = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {} }
		
		self.labels = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
		self.class_priors = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

	def fit(self, data):

		X = data["text"]
		Y = data["label"]
		m = len(X)

		#p prob
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

	def predict(self, data, text_normalised=False):
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

	def predict_random(self, test_data, text_normalised=False):
		for l in range(0, len(test_data)):
			i = randint(1, 5) #inclusive of 1 and 5
			predicted_labels.append(i)
		return predicted_labels

	def predict_majority(self, test_data, text_normalised=False):
		max_label = max(self.labels, key=self.labels.get) 
		for l in range(0, len(test_data)):
			predicted_labels.append(int(max_label))
		return predicted_labels
