# HMB while I write the preprocessing scripts for this part

# import numpy as np
# from sklearn.decomposition import PCA

# pca = PCA(n_components=50)
# pca.fit(X)

import os
import glob
import random
from PIL import Image
import pandas as pd

from sklearn.decomposition import PCA


from skimage import io
from skimage import color

import sys
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

def w():
	while True:
		pass

base_directory = "./data"
episodes = sorted(glob.glob(base_directory + "/*")) 

def get_frame_content(episode_path):
	frame_list = sorted(glob.glob(episode_path + "/*"))
	return {"frames": frame_list[:-1], "reward": frame_list[-1]}

content = get_frame_content(episodes[0])
f = content["frames"]
# print(len(f))
f = f[0:60]
reward_info = pd.read_csv(content["reward"], header=None)
reward_info = reward_info.values
# w();

def nouse():
	filepath_list = []
	for i in range(0, len(f)-6):
		filepath_list.append(f[i:i+7])
		# i+6 is the last frame

	# print((filepath_list))

	data_list = []
	for episode in filepath_list:
		b = set(random.sample(episode, 2))
		x = [i for i in episode if i not in b]
		data_list.append(x)
	print(data_list)

	for data in data_list:
		img = Image.open(data[0]).convert('L')
		wid, heig = img.size
		print(len(list(img.getdata())))
		print(wid, heig)


data = [] 
for d in f:
    arr = Image.open(d).convert('L')
    arr = np.array(list(arr.getdata()))
    data.append(arr)

arr = data[0]
for i in range(1, len(data)):
    arr = np.vstack((arr, data[i]))

data = arr
print(data.shape)

pca = PCA(n_components=50)
data = pca.fit_transform(data)
print(data.shape)
print(reward_info.shape)

"""
find X and Y now:
X as 5X50 = 250 features
Y as corresponding reward of t+3 of 7 frames window
"""

m = len(data)
X = []
Y = []
for i in range(0, m-9):
	# filepath_list.append(data[i:i+7])
	tmp = data[i:i+7]
	b = set(random.sample([i for i in range(7)], 2))
	x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
	x = x.flatten()
	# print(x.shape)
	X.append(x)
	Y.append(reward_info[i+9])
	# w();
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
	# i+6 is the last frame

from sklearn.svm import SVC
# kernel='rbf'
clf = SVC(kernel='linear')
clf.fit(X, Y)
acc = clf.score(X, Y)
print(acc)