import re
import json
import pprint
from string import punctuation
from nltk import word_tokenize

class Preprocess:
	"""get preprocessed data as data{"text":[], "label":[]}"""
	def __init__(self):
		with open('./dataset/sample.json') as f:
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


def count_frequency(tokens):
	"""given tokens, return the word count frequency"""
	
	count = {}
	for t in tokens:
		count[t] = count.get(t, 0.0) + 1.0
	return count


if __name__ == '__main__':
	processing = Preprocess()
	print(processing.data)



"""functions I will use later"""


	# def normalize(self):
	# 	lem = WordNetLemmatizer()
	# 	clean_tokens = self.tokens

	# 	for (w,t) in pos_tag(clean_tokens):
	# 		wt = t[0].lower()
			
	# 		wt = [wt if wt in ['a','r','n','v'] else None]
			
	# 		wnew = lem.lemmatize(w,wt) if wt else None
	# 		clean_tokens.remove(w)
	# 		clean_tokens.append(wnew)

	# 	self.tokens = clean_tokens
	# 	return clean_tokens

	# def stopwords_remove(self):		
	# 	rawtext = self.text
		
	# 	tokens = nltk.word_tokenize(tokens)
	# 	stopwords = set(stopwords.words('english'))

	# 	clean_tokens = [x for x in toks if not x in stopwords]
	# 	tokens = " ".join(clean_tokens)
	# 	return text