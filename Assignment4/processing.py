# HMB while I write the preprocessing scripts for this part

import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA

# np.set_printoptions(threshold=sys.maxsize)

def wait():
	while True:
		pass

class Preprocessing:
	def __init__(self):
		self.n_components = 5
		self.ratio = 0.9

		base_directory = "./data"
		self.episodes = sorted(glob.glob(base_directory + "/*"))[:50]
		self.num_of_episodes = len(self.episodes)

		num = int(self.num_of_episodes*self.ratio)
		self.train_episodes = self.episodes[:num] 
		self.test_episodes = self.episodes[num:]

		self.pca = PCA(n_components=self.n_components)

	def calculate_pca(self):
		"""
		Instruction: PCA should be calculated using all images
		of first 50 episodes
		"""
		no_episodes = 2
		print("[*] Running PCA over {} episodes".format(no_episodes))
		
		for X in self.raw_image_data_for_pca():
			if no_episodes < 0:
				break

			self.pca.fit(X)
			no_episodes -= 1


	def raw_image_data_for_pca(self):
		"""returns a generator for raw image data"""

		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[0:20]

			data = []
			for d in f:
				arr = Image.open(d).convert('L')
				arr = np.array(list(arr.getdata()))
				data.append(arr)

			arr = data[0]
			for i in range(1, len(data)):
				arr = np.vstack((arr, data[i]))

			yield arr


	def get_frame_content(self, episode_path):
		"""get content of all frames in an episode"""

		frame_list = sorted(glob.glob(episode_path + "/*"))
		return {"frames": frame_list[:-1], "reward": frame_list[-1]}


	def get_train_data_by_episode(self):
		"""return a generator over train data episode wise"""

		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[0:20]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = []
			for d in f:
				arr = Image.open(d).convert('L')
				arr = np.array(list(arr.getdata()))
				data.append(arr)

			arr = data[0]
			for i in range(1, len(data)):
				arr = np.vstack((arr, data[i]))

			data = self.pca.transform(arr)

			m = len(data)
			X = []
			Y = []
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.flatten()

				X.append(x)
				# 1st frame is reward
				Y.append(reward_info[i + 7])
			X = np.array(X)
			Y = np.array(Y)

			yield X, Y


	def get_test_data_by_episode(self):
		for i in range(len(self.test_episodes)):
			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[0:20]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = []
			for d in f:
				arr = Image.open(d).convert('L')
				arr = np.array(list(arr.getdata()))
				data.append(arr)

			arr = data[0]
			for i in range(1, len(data)):
				arr = np.vstack((arr, data[i]))

			data = self.pca.transform(arr)

			m = len(data)
			X = []
			Y = []
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.flatten()
				X.append(x)
				Y.append(reward_info[i + 7])

			X = np.array(X)
			Y = np.array(Y)

			yield X, Y


if __name__ == '__main__':
	p = Preprocessing()
