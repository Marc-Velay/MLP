#include "MLP.hpp"

using namespace std;

MLP::MLP(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO) {
        //Creating neuron list
        inputNeurons = new( double[nInput +1]);
        for(int i = 0; i < nInput; i++) {
                inputNeurons[i] = 0;
        }        
        inputNeurons[nInput] = 1;       //Bias neuron, weight static for entropy

        hiddenNeurons = new( double[nHidden +1]);
        for(int i = 0; i < nHidden; i++) {
                hiddenNeurons[i] = 0;
        }
        hiddenNeurons[nHidden] = 1;     //Bias neuron, weight static for entropy

        outputNeurons = new( double[nOutput +1]);
        for(int i = 0; i < nOutput; i++) {
                outputNeurons[i] = 0;
        }
        outputNeurons[nOutput] = 1;     //Bias neuron, weight static for entropy

        //Creating the weight table
        wInputHidden = new( double*[nInput +1]);
        for(int i = 0; i <= nInput; i++) {
                wInputHidden[i] = new( double[nHidden]);
                for(int j =0; j < nHidden; j++) {
                        wInputHidden[i][j] = 0;
                }
        }

        wHiddenOutput = new( double*[nHidden +1]);
        for(int i = 0; i <= nHidden; i++) {
                wHiddenOutput[i] = new( double[nOutput]);
                for(int j = 0; j < nOutput; j++) {
                        wHiddenOutput[i][j] = 0;
                }
        }

        initializeWeights();
}



MLP::~MLP()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	//delete weight storage
	for (int i=0; i <= nInput; i++) {
                delete[] wInputHidden[i];
        }
	delete[] wInputHidden;

	for (int i=0; i <= nHidden; i++) {
                delete[] wHiddenOutput[i];
        }
	delete[] wHiddenOutput;	
}


size_t MLP::strcpy_ss(char *d, size_t n, char const *s) {
    return snprintf(d, n, "%s", s);
}

bool MLP::loadWeights(char* file) {
        fstream inputFile;     
        inputFile.open(file, ios::in);  //Where the weights have been stored

        if( inputFile.is_open() ) {
                vector<double> weights;
                string line = "";

                while( !inputFile.eof() ) {     //Reading the whole file
                        getline(inputFile, line);

                        if( line.length() > 2 ) {
                                char* cstr = new char[line.size()+1];
                                char* t;
                                MLP::strcpy_ss(cstr, line.size() +1, line.c_str());

                                int i =0;
                                char* nextToken = NULL;
                                t=strtok_r(cstr,",", &nextToken);

                                while(t != NULL) {      //read the whole line
                                        weights.push_back( atof(t) );

                                        t = strtok_r(NULL, ",", &nextToken );   //move on the to next token
                                        i++;
                                }

                                delete[] cstr;  //free the used temp line holder
                        }
                }

                if((int)weights.size() != ( (nInput + 1) * nHidden + (nHidden + 1) * nOutput) ) {    //Check if the weights were properly loaded
                        cout << endl << "Error - Incorrect amount of weights were loaded from: " << file << endl;

                        inputFile.close();
                        return false;   //error, exit
                } else {        //set weights
                        int pos = 0;

                        for(int i = 0; i <= nInput; i++) {
                                for(int j = 0; j < nHidden; j++) {
                                        wInputHidden[i][j] = weights[pos++];
                                }
                        }

                        for(int i = 0; i <= nInput; i++) {
                                for(int j = 0; j < nHidden; j++) {
                                        wHiddenOutput[i][j] = weights[pos++];
                                }
                        }
                        //Weights successfuly loaded
                        cout << endl << "Neuron weights loaded successfuly from '" << file << "'" << endl;
                        inputFile.close();

                        return true;
                }
        } else {
                cout << endl << "Error - Could not open: " << file << endl;
        }
        return false;
}


bool MLP::saveWeights(char* file) {
        fstream outputFile;
        outputFile.open(file, ios::out);

        if( outputFile.is_open() ) {
                outputFile.precision(50);

                for(int i = 0; i <= nInput; i++) {
                        for(int j = 0; j < nHidden; j++) {
                                outputFile << wInputHidden[i][j] << ",";
                        }
                }


                for(int i = 0; i <= nHidden; i++) {
                        for(int j = 0; j < nOutput; j++) {
                                outputFile << wHiddenOutput[i][j]; 
                                if( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) {
                                        outputFile  << ",";
                                }
                        }
                }

                cout << endl << "Weigths saved to '" << file << "'" << endl;

                outputFile.close();

                return true;

        } else {
                cout << endl << "Error - Weight output file '" << file << "' could not be created: " << endl;
        }

        return false;
}



int* MLP::feedForwardPattern(double *pattern) {
        feedForward(pattern);

        int* results = new int[nOutput];
        for(int i = 0; i < nOutput; i++) {
                results[i] = clampOutput(outputNeurons[i]);
        }

        return results;
}



double MLP::getSetAccuracy( std::vector<dataEntry*>& set) {     //Get set accuracy
        double incorrectResults = 0;

        for( int i = 0; i < (int)set.size(); i++) {
                feedForward(set[i]->pattern);

                bool correctResultFlag = true;

                for(int j = 0; j < nOutput; j++) {
                        if( clampOutput(outputNeurons[j]) != set[i]->target[j]) {
                                correctResultFlag = false;
                        }
                }

                if(!correctResultFlag) {
                        incorrectResults++;
                }
        }

        return 100 - (incorrectResults/set.size() *100);
}



double MLP::getSetMSE( std::vector<dataEntry*>& set ) { //Calculate the Mean Square Error of the set
        double mse = 0;

        for (int i = 0; i < (int) set.size(); i++) {
                feedForward( set[i]->pattern );

                for( int j = 0; j < nOutput; j++) {
                        mse += pow((outputNeurons[j] - set[i]->target[j]), 2);
                }
        }

        return mse/(nOutput * set.size());
}



void MLP::initializeWeights() {
        double rHidden = 1/sqrt( (double) nInput);
        double rOutput = 1/sqrt( (double) nHidden);

        for(int i = 0; i <= nInput; i++) {
                for(int j = 0; j< nHidden; j++) {
                        wInputHidden[i][j] = ( ( (double)(rand()%100)+1)/100 *2 *rHidden) - rHidden;
                }
        }

        for(int i = 0; i <= nHidden; i++) {
                for(int j = 0; j< nOutput; j++) {
                        wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 *2 *rOutput) - rOutput;
                }
        }
}



inline double MLP::activationFunction( double x ) {
        //sigmoid function
	return 1/(1+exp(-x));
}



int MLP::clampOutput( double x ) {
        if( x < 0.1 ) return 0;
        else if( x > 0.9 ) return 1;
        else return -1;
}



void MLP::feedForward( double* pattern) {
        for(int i = 0; i < nInput; i++) {
                inputNeurons[i] = pattern[i];
        }

        for(int i = 0; i < nHidden; i++) {
                hiddenNeurons[i] = 0;

                for(int j = 0; j <= nInput; j++) {
                        hiddenNeurons[i] += inputNeurons[j] * wInputHidden[j][i];
                }

                hiddenNeurons[i] = activationFunction( hiddenNeurons[i] );
        }

        for(int i = 0; i < nOutput; i++) {
                outputNeurons[i] = 0;

                for(int j = 0; j < nHidden; j++) {
                        outputNeurons[i] += hiddenNeurons[j] * wHiddenOutput[j][i];
                }

                outputNeurons[i] = activationFunction( outputNeurons[i]);
        }
        
}
