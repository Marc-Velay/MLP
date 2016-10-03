#include <cstdio>
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <math.h>
#include "dataReader.hpp"

using namespace std;
class MLPTrainer;

class MLP{
        //number of neurons
    int nInput, nHidden, nOutput;

    //neurons
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons;

    //weights
	double** wInputHidden;
	double** wHiddenOutput;

    public: 
        MLP(int numInput, int numHidden, int numOutput);
	    ~MLP();

        
        bool loadWeights(char* inputFilename);
        bool saveWeights(char* outputFilename);
        int* feedForwardPattern( double* pattern );
        double getSetAccuracy( std::vector<dataEntry*>& set );
        double getSetMSE( std::vector<dataEntry*>& set );

    private:
        void initializeWeights();
        inline double activationFunction( double x );
        inline int clampOutput( double x );
        void feedForward( double* pattern );
        size_t strcpy_ss(char *d, size_t n, char const *s);

        friend class MLPTrainer;
};
