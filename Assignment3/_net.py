# Hold my beer while write these support functions

import pandas as pd 
import numpy as np

def convert_to_one_hot(input_file, output_file):
	traindata = pd.read_csv(input_file)

	final_data=[]
	Y_data = []
	for k in range(len(traindata)):
		onehot=[]
		for i in range(0, 10):
			if i%2==0:
				tmp = [0]*4
				tmp[traindata.iloc[k, i]-1] = 1
				onehot = onehot + tmp
			else:
				tmp = [0]*13
				tmp[traindata.iloc[k, i]-1] = 1
				onehot = onehot + tmp

		tmp = [0]*10
		tmp[traindata.iloc[k, 10]] = 1
		onehot = onehot + tmp

		final_data.append(onehot)

	a = np.asarray(final_data).astype(int)
	np.savetxt(output_file, a, delimiter=",",fmt="%1.0f")
	print("[*] Saved as {}!\n".format(output_file))


if __name__ == '__main__':
	import sys

	training_file = str(sys.argv[1])
	testing_file = str(sys.argv[2])
	one_hot_file_train = str(sys.argv[3])
	one_hot_file_test = str(sys.argv[4])

	convert_to_one_hot(training_file, one_hot_file_train)
	convert_to_one_hot(testing_file, one_hot_file_test)