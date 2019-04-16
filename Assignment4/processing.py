# HMB while I write the preprocessing scripts for this part

import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA

from skimage import io
from skimage import color

# np.set_printoptions(threshold=sys.maxsize)


def wait():
    while True:
        pass


class Preprocessing:
    def __init__(self):
        base_directory = "./data"
        self.episodes = sorted(glob.glob(base_directory + "/*"))

    def get_frame_content(self, episode_path):
        frame_list = sorted(glob.glob(episode_path + "/*"))
        return {"frames": frame_list[:-1], "reward": frame_list[-1]}

    def get_train_data_by_episode(self):
        for i in range(len(self.episodes)):
            content = self.get_frame_content(self.episodes[i])
            f = content["frames"]
            f = f[0:60]

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

            data = arr
            # print(data.shape)
            """ Apply PCA on all frames of a episode: (no-examples) X (210*160) """
            pca = PCA(n_components=50)
            data = pca.fit_transform(data)
            # print(data.shape)
            # print(reward_info.shape)
            """
			find X and Y now:
			X as 5X50 = 250 features
			Y as corresponding reward of t+3 of 7 frames window
			"""
            m = len(data)
            X = []
            Y = []
            for i in range(0, m - 9):
                # filepath_list.append(data[i:i+7])
                tmp = data[i:i + 7]
                b = set(random.sample([i for i in range(7)], 2))
                x = np.array([tmp[i] for i in range(len(tmp)) if i not in b])
                x = x.flatten()
                # print(x.shape)
                X.append(x)
                # 3rd frame is reward
                Y.append(reward_info[i + 9])
                # w();
            X = np.array(X)
            Y = np.array(Y)

            # print(X.shape)
            # print(Y.shape)

            yield X, Y


if __name__ == '__main__':
    p = Preprocessing()
    episode = 1

    # get data by generator
    for X, Y in p.get_train_data_by_episode():
        print("[*] Data episode {}".format(episode))
        print(X.shape)
        print(Y.shape)
        episode += 1
        break
