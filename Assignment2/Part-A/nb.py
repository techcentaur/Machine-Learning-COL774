import math
from random import randint

import preprocessing


class NaiveBayes:
	def __init__(self, png):
		self.png = png

		self.vocabulary = set()
		self.class_words = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {} }
		
		self.labels = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
		self.class_priors = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

	def fit(self, data):
		X = data["text"]
		Y = data["label"]
		m = len(X)

		print("[.] Number of training examples: {}".format(m))

		for label in Y:
			self.labels[str(label)] += 1
		for key in self.class_priors:
			self.class_priors[key] = math.log(self.labels[key]/m) 


		# iteration over data space
		for x, y in zip(X, Y):
			freqs = preprocessing.count_frequency(x)

			for word, count in freqs.items():
				if word not in self.vocabulary:
					self.vocabulary.add(word)
				if word not in self.class_words[str(y)]:
					self.class_words[str(y)][word] = 0.0

				self.class_words[str(y)][word] += count

		print(self.labels)

	def predict(self, test_data):
		predicted_labels = []

		for x in test_data:
			good_data = self.png.normalise_data(x)
			counts = preprocessing.count_frequency(good_data)

			score = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }
			log_scores = { "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 }

			for word, count in counts.items():
				if word not in self.vocabulary:
					continue

				# Add Laplace smoothing here
				for key in self.class_words:
					log_scores[key] = math.log((self.class_words[key].get(word, 0.0) + 1)/(len(self.class_words[key]) + len(self.vocabulary)))
					score[key] += log_scores[key]
		
			# add class prior probs in log space
			for key in self.class_words:      
				score[key] += self.class_priors[key]

			prediction = max(score, key=score.get)
			predicted_labels.append(int(prediction))

		return predicted_labels

	def predict_random(self, test_data):
		predicted_labels = []

		for l in len(test_data):
			i = randint(1, 5)
			predicted_labels.append(i)

		return predicted_labels


if __name__ == '__main__':
	png = preprocessing.Preprocess()
	
	model = NaiveBayes(png)
	model.fit(png.data)

	# predicted_labels = model.predict(png.data["text"])
	# accuracy = sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == actual_labels[i]]) / float(len(predicted_labels))

	# print("Accuracy: {0:.4f}".format(accuracy))