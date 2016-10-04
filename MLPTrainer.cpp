#include <iostream>
#include <fstream>
#include <math.h>
#include "MLPTrainer.hpp"

using namespace std;

MLPTrainer::MLPTrainer( MLP *untrainedNetwork) : NN(untrainedNetwork), learningRate(LEARNING_RATE), momentum(MOMENTUM), epoch(0), maxEpochs(MAX_EPOCHS), desiredAccuracy(DESIRED_ACCURACY),trainingSetAccuracy(0), validationSetAccuracy(0), generalizationSetAccuracy(0), trainingSetMSE(0), validationSetMSE(0),generalizationSetMSE(0)	 
{
    cout << "MLPTrainer constructor" << endl;
    deltaInputHidden = new( double*[NN->nInput +1]);
    for(int i = 0; i <= NN->nInput; i++) {
        deltaInputHidden[i] = new(double[NN->nHidden]);
        for(int j = 0; j < NN->nHidden; j++) {
            deltaInputHidden[i][j] = 0;
        }
    }

    deltaHiddenOutput = new( double*[NN->nHidden +1]);
    for(int i = 0; i <= NN->nHidden; i++) {
        deltaHiddenOutput[i] = new(double[NN->nOutput]);
        for(int j = 0; j < NN->nOutput; j++) {
            deltaHiddenOutput[i][j] = 0;
        }
    }

    hiddenErrorGradients = new(double[NN->nHidden +1]);
    for(int i = 0; i <= NN->nHidden; i++) {
        hiddenErrorGradients[i] = 0;
    }

    outputErrorGradients = new(double[NN->nOutput +1]);
    for(int i = 0; i <= NN->nOutput; i++) {
        outputErrorGradients[i] = 0;
    }
}


void MLPTrainer::setTrainingParameters(double lR, double m) {
    learningRate = lR;
    momentum = m;
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
        cout << "trainingSetAccuracy: " << trainingSetAccuracy << ", generalizationSetAccuracy: " << generalizationSetAccuracy << endl;
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
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl << endl;	
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

    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
    trainingSetMSE = mse / ( NN->nOutput * trainingSet.size() );
} 


void MLPTrainer::backpropagate( double* desiredOutputs) {
    for(int i = 0; i < NN->nOutput; i++) {
        outputErrorGradients[i] = getOutputErrorGradient( desiredOutputs[i], NN->outputNeurons[i]);
        for(int j = 0; j <= NN->nHidden; j++) {
            deltaHiddenOutput[j][i] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[i] + momentum * deltaHiddenOutput[j][i];            
        }
    }
    for(int i = 0; i < NN->nHidden; i++) {
        hiddenErrorGradients[i] = getHiddenErrorGradient( i );
        for(int j = 0; j <= NN->nInput; j++) {
            deltaInputHidden[j][i] = learningRate * NN->inputNeurons[j] * hiddenErrorGradients[i] + momentum * deltaInputHidden[j][i];
        }
    }
    updateWeights();
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
						
		}
	}
	
	//hidden -> output weights
	for (int i = 0; i <= NN->nHidden; i++)
	{
		for (int j = 0; j < NN->nOutput; j++) 
		{					
			//update weight
			NN->wHiddenOutput[i][j] += deltaHiddenOutput[i][j];
			
		}
	}
}


int main(void){
    printf("Starting training !\n\n");
    srand( (unsigned int) time(0) );

    dataReader d;
    d.loadDataFile("k-meansData.csv",2,1);
    d.setCreationApproach();
    cout << "Initialised dataset and approach" << endl;
    //create neural network
	MLP mlp(2,5,1);
    cout << "created MLP" << endl;

    MLPTrainer mlpTrainer( &mlp );
    cout << "created trainer" << endl;
	mlpTrainer.setTrainingParameters(0.001, 0.9);    //learning rate and momentum
	mlpTrainer.setStoppingConditions(150, 90);  //nb Epochs, %desired accuracy
	mlpTrainer.enableLogging("log.csv", 5);     //log every 5 epochs
    cout << "created trainer parameters" << endl;

    //train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
        cout << "training network on datasets" << endl;
		mlpTrainer.trainNetwork( d.getTrainingDataSet() );
	}

    //save the weights
    char * file = "weights.csv";
	mlp.saveWeights(file);


    cout << endl << endl << "Finished training, weights saved to: 'weights.csv'" << endl;
    return 0;           
}