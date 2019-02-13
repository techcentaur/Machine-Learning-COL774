#!/bin/bash
if [ "$1" == "1" ]
then
	python3 linear_regression.py $2 $3 $4 $5
elif [ "$1" == "2" ]
then
	python3 locally_weighted_linear_regression.py $2 $3 $4
elif [ "$1" == "3" ]
then
	python3 logistic_regression.py $2 $3
elif [ "$1" == "4" ]
then
	python3 gda.py $2 $3 $4
fi
