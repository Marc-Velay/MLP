#! /bin/bash

rm log.csv
rm weights.csv
touch log.csv
touch weights.csv
sh createDataSet.sh
make clean
make
