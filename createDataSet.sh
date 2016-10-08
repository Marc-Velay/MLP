#! /bin/bash
rm k-meansData.csv
touch k-meansData.csv
g++ -Wall -Wextra -o createDataSet createDataSet.cpp
./createDataSet
rm createDataSet
echo "Finished creating the data set\n"