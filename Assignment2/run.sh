#!/bin/bash

# This should train classifier using train data
#and report accuracy and macro f-score on test data

if [ "$1" == "1" ]
then
	if ["$4" == "a"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "b"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "c"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "d"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "e"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "f"]
		python3 nb.py $2 $3 $4
	elif ["$4" == "g"]
		python3 nb.py $2 $3 $4
elif [ "$1" == "2" ]
then
	if ["$4" == "0"]
		if ["$5" == "a"]
			python3 svm_python.py $2 $3 $4 $5
		elif ["$5" == "b"]
			python3 svm_python.py $2 $3 $4 $5
		elif ["$5" == "c"]
			python3 svm_python.py $2 $3 $4 $5
		fi
	elif ["$4" == "1"]
		if ["$5" == "a"]
			python3 multisvm.py $2 $3 $4 $5
		elif ["$5" == "b"]
			python3 multisvm.py $2 $3 $4 $5
		elif ["$5" == "c"]
			python3 multisvm.py $2 $3 $4 $5
		elif ["$5" == "d"]
			python3 multisvm.py $2 $3 $4 $5
		fi
	fi
fi
