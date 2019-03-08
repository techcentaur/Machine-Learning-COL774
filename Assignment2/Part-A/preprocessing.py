import re
import ast
import json
import pprint

from string import punctuation
from nltk import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

class Preprocess:
	"""get preprocessed data as data{"text":[], "label":[]}"""
	def __init__(self, file_path, verbose=False, stem=False, stopwords=False):
		self.verbose = verbose
		self.stem = stem
		self.stopwords = stopwords

		with open(file_path) as f:
			raw_data = f.readlines()

		raw_data = [ast.literal_eval(x.strip()) for x in raw_data]

		data = {"text": [], "label": []}
		for raw_datum in raw_data:
			data["text"].append(self.normalise_data(raw_datum["text"]))
			data["label"].append(int(raw_datum["stars"]))

		self.data = data
		self.num_examples = len(self.data["text"])
		
		if self.verbose:
			print("[#] Data preprocessed! ")
			print("[>] Number of examples {}".format(self.num_examples))
			print("[>] Sample example: {} \n {}\n".format(self.data["text"][0], self.data["label"][0]))


	def train_and_test(self, ratio=0.8):
		"""Divide `self.data` into train and test based on ratio"""

		train = {"text": [], "label": []}
		test = {"text": [], "label": []}

		partition = int(self.num_examples*0.8)

		# partition of dataset into train and test
		train["text"] = self.data["text"][:partition]
		train["label"] = self.data["label"][:partition]

		test["text"] = self.data["text"][partition:]
		test["label"] = self.data["label"][partition:]

		if self.verbose:
			print("[!] Data partitioned into train and test with ratio {}".format(ratio))

		return {"train": train, "test": test}

	def read_file_as_dict(file_path):
		"""if file in a dict: Read from here"""

		with open(file_path) as f:
		   raw_data = json.load(f)

		data = {"text": [], "label": []}
		for key in raw_data:
			data["text"].append(self.normalise_data(raw_data[key]["text"]))
			data["label"].append(raw_data[key]["stars"])

		self.data = data

	def normalise_data(self, text):
		"""normalise a give string"""

		text = self.apostrophe_normalisation(text)
		tokens = self.punctuation_remove(word_tokenize(text))
		tokens = [x.lower() for x in tokens]
		if self.stopwords:
			tokens = self.stopwords_removal(tokens)
		if self.stem:
			tokens = self.stemming(tokens)

		return tokens

	def apostrophe_normalisation(self, text):
		"""takes a string, and normalise strings in it as per the rules, returns a string"""

		raw = text
		raw = re.sub(r"n\'t", " not", raw)
		raw = re.sub(r"\'re", " are", raw)
		raw = re.sub(r"\'s", " is", raw)
		raw = re.sub(r"\'d", " would", raw)
		raw = re.sub(r"\'ll", " will", raw)
		raw = re.sub(r"\'t", " not", raw)
		raw = re.sub(r"\'ve", " have", raw)
		raw = re.sub(r"\'m", " am", raw)
		raw = re.sub(r"\'cause", "because", raw)
		raw = re.sub(r"\'Cause", "Because", raw)
		return raw


	def punctuation_remove(self, tokens):
		"""given a list of tokens, remove the punctuations, and return tokens"""

		punctuation_list = list(punctuation)
		for i in tokens:
			if i in punctuation_list:
				tokens.remove(i)
		return tokens

	def stemming(self, tokens):
		"""stem words from a list of tokens"""
		ps = PorterStemmer() 
		tokens = [ps.stem(tok) for tok in tokens]
		return tokens

	def stopwords_removal(self, tokens):				
		"""remove stopwords from tokens"""

		sw = set(stopwords.words('english'))
		tokens = [x for x in tokens if not x in sw]
		return tokens


def count_frequency(tokens):
	"""given tokens, return the word count frequency"""
	
	count = {}
	for t in tokens:
		count[t] = count.get(t, 0.0) + 1.0
	return count


if __name__ == '__main__':
	processing = Preprocess()
	print(processing.data)


