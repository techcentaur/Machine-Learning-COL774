# HMB while I write the preprocessing scripts for this part

"""
Usage
Preprocessing: Class
Methods:
calculate_pca(): To calculate pca over first 50 episodes
get_train_data_by_episode(): Get train data generator over train episodes
get_test_data_by_episode(): Get test data generator over test episodes
"""
import os
import sys
import csv
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import time


from sklearn.decomposition import PCA
import pickle
# np.set_printoptions(threshold=sys.maxsize)

def wait():
	while True:
		pass

class Preprocessing:
	def __init__(self):
		self.n_components = 50
		self.ratio = 0.9

		base_directory = "./data"
		self.episodes = sorted(glob.glob(base_directory + "/*"))[:200]
		self.num_of_episodes = len(self.episodes)

		num = int(self.num_of_episodes*self.ratio)
		self.train_episodes = self.episodes[:num] 
		self.test_episodes = self.episodes[num:]

		self.pca = PCA(n_components=self.n_components)
		self.comp_bigdata = False

	def calculate_pca(self):
		no_episodes = 50
		tmp = no_episodes
		if os.path.exists("./pca_saved_{}.pkl".format(tmp)):
			with open("./pca_saved_{}.pkl".format(tmp), 'rb') as f:
				print("[*] PCA-model already calculated - Loaded...")
				self.pca = pickle.load(f)
			return True

		print("[*] Running PCA over {} episodes".format(tmp))

		first = True
		X_train = None		
		for X in self.raw_image_data_for_pca():
			if no_episodes < 0:
				break
			print("[{}] ".format(no_episodes), end=' ')

			if first:
				X_train = X
				first = False
				no_episodes -= 1
				continue
			print(X.shape)
			X_train = np.concatenate((X_train, X), axis=0)
			no_episodes -= 1

		print(X_train.shape)
		self.pca.fit(X_train)
		# wait()
		with open("./pca_saved_{}.pkl".format(tmp), 'wb') as f:
			pickle.dump(self.pca, f)
			print("[*] PCA-computed model saved!")

		return True

	def raw_image_data_for_pca(self):
		"""returns a generator for raw image data"""

		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			# f = f[0:200]

			data = None
			first = True
			for d in f:
				arr = Image.open(d).convert('L')
				arr = np.array(list(arr.getdata()))
				# print(arr)
				if first:
					data = arr
					first = False
					continue
				data = np.vstack((data, arr))

			yield data


	def get_frame_content(self, episode_path):
		"""get content of all frames in an episode"""

		frame_list = sorted(glob.glob(episode_path + "/*"))
		return {"frames": frame_list[:-1], "reward": frame_list[-1]}


	def get_train_data_by_episode(self):
		print("[-] Total Number of Episodes: {}".format(len(self.train_episodes)))
		
		num = 0
		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[:100]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 250))
		biglabel = np.ndarray(shape=(num, 1))

		prev=0
		for i in range(len(self.train_episodes)):
			print("[.] Episode {}".format(i))

			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[:100]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 33600))
			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).convert('L'))).flatten()
				data[i] = arr
			data = self.pca.transform(data)
			m = len(data)
			X = np.ndarray(shape=(m-7, 250))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.flatten()
				X[i] = x
				Y[i] = reward_info[i + 7]
				# reward frame
			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel

	def get_test_data_by_episode(self):
		"""generator for test data"""

		print(len(self.test_episodes))
		num = 0
		for i in range(len(self.test_episodes)):
			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[:100]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 250))
		biglabel = np.ndarray(shape=(num, 1))

		prev=0
		for i in range(len(self.test_episodes)):
			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[:100]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 33600))
			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).convert('L'))).flatten()
				data[i] = arr
			data = self.pca.transform(data)
			m = len(data)
			X = np.ndarray(shape=(m-7, 250))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.flatten()
				X[i] = x
				Y[i] = reward_info[i + 7]
				# reward frame
			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel


	def get_train_data_with_rgbchannel(self):
		print("[-] Total Number of Episodes: {}".format(len(self.train_episodes)))
		
		num = 0
		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[:100]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 15, 210, 160))
		biglabel = np.ndarray(shape=(num, 1))

		prev = 0
		for i in range(len(self.train_episodes)):
			print("[.] Episode {}".format(i))

			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[0:100]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 3, 210, 160))

			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).resize((210, 160))).T).reshape(-1, 210, 160)
				data[i] = arr

			m = len(data)
			X = np.ndarray(shape=(m-7, 15, 210, 160))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.reshape(x.shape[:-4] + (-1, 210, 160))

				X[i] = x
				Y[i] = reward_info[i + 7]

			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel

	def get_test_data_with_rgbchannel(self):
		print("[-] Total Number of Episodes: {}".format(len(self.test_episodes)))
		
		num = 0
		for i in range(len(self.test_episodes)):
			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[:100]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 15, 210, 160))
		biglabel = np.ndarray(shape=(num, 1))

		prev = 0
		for i in range(len(self.test_episodes)):
			print("[.] Episode {}".format(i))

			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[0:100]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 3, 210, 160))

			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).resize((210, 160))).T).reshape(-1, 210, 160)
				data[i] = arr;

			m = len(data)
			X = np.ndarray(shape=(m-7, 15, 210, 160))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.reshape(x.shape[:-4] + (-1, 210, 160))

				X[i] = x
				Y[i] = reward_info[i + 7]

			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel


	def get_train_data_with_rgbchannel_for_comp(self):
		print("[-] Total Number of Episodes: {}".format(len(self.train_episodes)))
		
		num = 0
		for i in range(len(self.train_episodes)):
			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[:500]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 15, 40, 40))
		biglabel = np.ndarray(shape=(num, 1))

		prev = 0
		for i in range(len(self.train_episodes)):
			print("[.] Episode {}".format(i))

			content = self.get_frame_content(self.train_episodes[i])
			f = content["frames"]
			f = f[0:500]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 3, 40, 40))

			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).resize((40, 40))).T).reshape(-1, 40, 40)
				data[i] = arr;

			m = len(data)
			X = np.ndarray(shape=(m-7, 15, 40, 40))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.reshape(x.shape[:-4] + (-1, 40, 40))

				X[i] = x
				Y[i] = reward_info[i + 7]

			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel

	def get_test_data_with_rgbchannel_for_comp(self):
		print("[-] Total Number of Episodes: {}".format(len(self.train_episodes)))
		
		num = 0
		for i in range(len(self.test_episodes)):
			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[:500]
			num += (len(f)-7)

		bigdata = np.ndarray(shape=(num, 15, 40, 40))
		biglabel = np.ndarray(shape=(num, 1))

		prev = 0
		for i in range(len(self.test_episodes)):
			print("[.] Episode {}".format(i))

			content = self.get_frame_content(self.test_episodes[i])
			f = content["frames"]
			f = f[0:500]

			reward_info = pd.read_csv(content["reward"], header=None)
			reward_info = reward_info.values

			data = np.ndarray(shape=(len(f), 3, 40, 40))

			for i in range(0, len(f)):
				arr = (np.asarray(Image.open(f[i]).resize((40, 40))).T).reshape(-1, 40, 40)
				data[i] = arr;

			m = len(data)
			X = np.ndarray(shape=(m-7, 15, 40, 40))
			Y = np.ndarray(shape=(m-7, 1))
			for i in range(0, m - 7):
				tmp = data[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.reshape(x.shape[:-4] + (-1, 40, 40))

				X[i] = x
				Y[i] = reward_info[i + 7]

			assert(m==(len(f)))
			bigdata[prev: prev+len(f)-7] = X
			biglabel[prev: prev+len(f)-7] = Y
			prev += len(f)-7

		return bigdata, biglabel




	def get_test_data_for_comp(self):
		direc = "./test_dataset"

		os.chdir(direc)
		csvData = [['id', 'Prediction']]
		l = 30909 + 1
		second = True
		bigdata = np.ndarray(shape=(30910, 15, 210, 160))
		for i in range(0,l):
			di = "0"*(8-len(str(i))) + str(i)
			os.chdir(di)
			imgs = os.listdir()
			first = True
			data = np.ndarray(shape=(5,3,210,160))
			for i in range(0, len(imgs)):
				arr = (np.asarray(Image.open(imgs[i])).T).reshape(-1, 210, 160)
				data[i] = arr

			data = data.reshape(data.shape[:-4] + (-1, 210, 160))
			bigdata[i] = data

			os.chdir("..")
		print(bigdata.shape)

		return bigdata

	def get_test_data_for_comp2(self):
		direc = "./test_dataset"

		os.chdir(direc)
		csvData = [['id', 'Prediction']]
		l = 30909 + 1
		second = True
		bigdata = np.ndarray(shape=(30910, 250))
		prev=0

		print("reading test data")
		for i in range(0,l):
			di = "0"*(8-len(str(i))) + str(i)
			os.chdir(di)
			imgs = os.listdir()
			first = True
			data = np.ndarray(shape=(len(f), 33600))
			data1 = np.ndarray(shape=(len(f), 50))
			for i in range(0, len(imgs)):
				arr = (np.asarray(Image.open(f[i]).convert('L'))).flatten()
				data[i] = arr

			data1 = self.pca.transform(data)

			m = len(data1)
			X = np.ndarray(shape=(m-7, 250))

			for i in range(0, m - 7):
				tmp = data1[i:i + 7]
				b = set(random.sample([i for i in range(6)], 2))
				x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
				x = x.flatten()
				X[i] = x
				# reward frame

			bigdata[prev: prev+m-7] = X
			prev += m-7

			os.chdir("..")
			bigdata[i] = data

		return bigdata


if __name__ == '__main__':
	p = Preprocessing()
