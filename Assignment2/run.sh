#!/bin/bash

# This should train classifier using train data
#and report accuracy and macro f-score on test data

# Here, 'binary_or_multi_class' is 0 for binary
# classification and 1 for multi-class. 
# 'part_num' is part number which can be a-c for binary
# classification and a-d for multi-class.

if [ "$1" == "1" ]
then
	python3 ./Part-A/nb.py $2 $3 $4
elif [ "$1" == "2" ]
then
	if [ "$4" == "0" ]
	then
		python3 ./Part-B/svm_python.py $2 $3 $5
	elif [ "$4" == "1" ]
	then
		python3 ./Part-B/multisvm.py $2 $3 $5
	fi
fi
