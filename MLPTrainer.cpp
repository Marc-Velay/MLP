#include <iostream>
#include <fstream>
#include <math.h>
#include "MLPTrainer.hpp"

using namespace std;

MLPTrainer::MLPTrainer( MLP *mlp) : NN(mlp), learningRate(LEARNING_RATE), epoch(0), momentum(MOMENTUM), maxEpochs(MAX_EPOCHS), desiredAccuracy(DESIRED_ACCURACY),trainingSetAccuracy(0), useBatch(false), validationSetAccuracy(0), generalizationSetAccuracy(0), trainingSetMSE(0), validationSetMSE(0),generalizationSetMSE(0)	 
{
    deltaInputHidden = new( double*[NN->nHidden]);
    for(int i = 0; i <= NN->nInput; i++) {
        deltaInputHidden[i] = new(double[NN->nHidden]);
        for(int j = 0; j < NN->nHidden; j++) {
            deltaInputHidden[i][j] = 0;
        }
    }

    deltaHiddenOutput = new(double*[NN->nOutput]);
    for(int i = 0; i <= NN->nHidden; i++) {
        deltaInputHidden[i] = new(double[NN->nOutput]);
        for(int j = 0; j < NN->nOutput; j++) {
            deltaHiddenOutput[i][j] = 0;
        }
    }

    hiddenErrorGradients = new(double[NN->nHidden +1]);
    for(int i = 0; i < NN->nHidden; i++) {
        hiddenErrorGradients[i] = 0;
    }

    outputErrorGradients = new(double[NN->nOutput +1]);
    for(int i = 0; i < NN->nOutput; i++) {
        outputErrorGradients[i] = 0;
    }
}


void MLPTrainer::setTrainingParameters(double lR, double m, bool batch) {
    learningRate = lR;
    momentum = m;
    useBatch = batch;
}


void MLPTrainer::setStoppingConditions(int mEpochs, double dAccuracy) {
    maxEpochs = mEpochs;
    desiredAccuracy = dAccuracy;
}

void MLPTrainer::enableLogging(const char* file, int resolution = 1) {
    if(!logFile.is_open()) {
        logFile.open(file, ios::out);

        if(logFile.is_open()) {
            logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;

            loggingEnabled = true;

            logResolution = resolution;
            lastEpochLogged = -resolution;
        }
    }
}



inline double MLPTrainer::getOutputErrorGradient( double desiredValue, double outputValue) {
    return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}


double MLPTrainer::getHiddenErrorGradient( int j ) {
    double weightedSum = 0;
    for(int i = 0; i < NN->nOutput; i++) {
        weightedSum += NN->wHiddenOutput[j][i] * outputErrorGradients[i];
    }

    return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}


void MLPTrainer::trainNetwork( trainingDataSet* tSet ) {
    epoch = 0;
    lastEpochLogged = -logResolution;

    //train using training dataset, generalizationSet for testing

    while( (trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs) {
        double previousTAccuracy = trainingSetAccuracy;
        double previousGAccuracy = generalizationSetAccuracy;

        runTrainingEpoch(tSet->trainingSet);

        generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet);
        generalizationSetMSE = NN->getSetMSE( tSet->generalizationSet);

        if(loggingEnabled && logFile.is_open() && (epoch - lastEpochLogged == logResolution)) {
            logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
            lastEpochLogged = epoch;
        }

        if( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) {
            cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;	
        }

        epoch++;
    }

    validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
	validationSetMSE = NN->getSetMSE(tSet->validationSet);

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
	logFile << "Training Complete: - > Elapsed Epochs: " << epoch  << endl;			
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
	cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}


void MLPTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet) {
    double incorrectPatterns = 0;
    double mse = 0;

    for(int i = 0; i < (int)trainingSet.size(); i++) {
        NN->feedForward(trainingSet[i]->pattern);
        backpropagate(trainingSet[i]->target);

        bool patternCorrect = true;

        for(int j = 0; j < NN->nOutput; j++) {
            if(NN->clampOutput(NN->outputNeurons[j]) != trainingSet[i]->target[j] ) {
                patternCorrect = false;
            }
            mse += pow((NN->outputNeurons[j] - trainingSet[i]->target[j]), 2);
        }
        if(!patternCorrect) incorrectPatterns++;
    }
    if(useBatch) updateWeights();

    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
    trainingSetMSE = mse / ( NN->nOutput * trainingSet.size() );
} 


void MLPTrainer::backpropagate( double* desiredOutputs) {
    for(int i = 0; i < NN->nOutput; i++) {
        outputErrorGradients[i] = getOutputErrorGradient( desiredOutputs[i], NN->outputNeurons[i]);

        for(int j = 0; j <= NN->nHidden; j++) {
            if(!useBatch) {
                deltaHiddenOutput[j][i] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[i] + momentum * deltaHiddenOutput[j][i];
            } else {
                deltaHiddenOutput[j][i] += learningRate * NN->hiddenNeurons[j] * outputErrorGradients[i];
            }
        }
    }

    for(int i = 0; i < NN->nHidden; i++) {
        hiddenErrorGradients[i] = getHiddenErrorGradient( i );

        for(int j = 0; j <= NN->nInput; j++) {
            if(!useBatch) {
                deltaInputHidden[j][i] = learningRate * NN->inputNeurons[j] * hiddenErrorGradients[i] + momentum * deltaInputHidden[j][i];
            } else {
                deltaHiddenOutput[j][i] += learningRate * NN->inputNeurons[j] * hiddenErrorGradients[i];
            }
        }
    }
    if(!useBatch) updateWeights();
}


void MLPTrainer::updateWeights()
{
	//input -> hidden weights
	for (int i = 0; i <= NN->nInput; i++)
	{
		for (int j = 0; j < NN->nHidden; j++) 
		{
			//update weight
			NN->wInputHidden[i][j] += deltaInputHidden[i][j];	
			
			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch) deltaInputHidden[i][j] = 0;				
		}
	}
	
	//hidden -> output weights
	for (int i = 0; i <= NN->nHidden; i++)
	{
		for (int j = 0; j < NN->nOutput; j++) 
		{					
			//update weight
			NN->wHiddenOutput[i][j] += deltaHiddenOutput[i][j];
			
			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHiddenOutput[i][j] = 0;
		}
	}
}


int main(void){
            printf("Hello World !\n\n");
            
            return 0;           
}