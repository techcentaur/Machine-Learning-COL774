"""
Main file: Call all functions from here.

"""

from svm_a import ModelSVM
from cnn import ModelCNN
import part2
# from neural_b import ModelNN

from processing import Preprocessing

def wait():
	while True:
		pass

def svm_run():
	# kernel = "linear"
	kernel = "rbf"
	modelS = ModelSVM(kernel=kernel)
	modelS.fit(p, verbose=True)
	print(modelS.score(p))


def cnn_run():
	param = {
	"num_epoches": 1,
	"batch_size": 20,
	"num_classes": 2,
	"learning_rate": 0.001
	}

	modelcnn = ModelCNN(param=param)
	modelcnn.fit(p)
	modelcnn.score(p)

def cnn_run2():
	param = {
	"num_epoches": 20,
	"batch_size": 200,
	"num_classes": 2,
	"learning_rate": 0.001
	}

	modelcnn = part2.ModelCNN(param=param)
	modelcnn.fit(p)
	modelcnn.score(p)

if __name__ == '__main__':
	import sys

	print("[*] Preprocessing!")
	p = Preprocessing()
	p.calculate_pca()

	# svm_run()
	# cnn_run()
	cnn_run2()
