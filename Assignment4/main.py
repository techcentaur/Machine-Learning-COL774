"""
Main file: Call all functions from here.

"""

from svm_a import ModelSVM
from cnn import ModelCNN
# from neural_b import ModelNN

from processing import Preprocessing

def wait():
	while True:
		pass

def svm_run():
	kernel = "linear"
	# kernel = "rbf"
	modelS = ModelSVM(kernel=kernel)
	modelS.fit(p, verbose=True)
	print(modelS.score(p))


def cnn_run():
	param = {
	"num_epoches": 2,
	"batch_size": 64,
	"num_classes": 2,
	"learning_rate": 0.0001
	}

	modelcnn = ModelCNN(param=param)
	modelcnn.fit(p)
	modelcnn.score(p)

if __name__ == '__main__':
	import sys

	p = Preprocessing()
	print("[*] Preprocessing!")
	# p.get_train_data_with_rgbchannel()
	print("[*] Calculating PCA!")
	p.calculate_pca()
	print("[*] PCA Done!")

	cnn_run()

