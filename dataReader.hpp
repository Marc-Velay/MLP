#include <vector>
#include <string>

class dataEntry         //stores a data item
{
public:	
	
	double* pattern;	//input patterns
	double* target;		//target result

	dataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}

};


class trainingDataSet       //Training Sets Storage - stores shortcuts to data items
{
public:

	std::vector<dataEntry*> trainingSet;
	std::vector<dataEntry*> generalizationSet;
	std::vector<dataEntry*> validationSet;

	trainingDataSet(){}
	
	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

//dataset retrieval approach enum
enum { NONE, STATIC};

//data reader class
class dataReader
{
	
//private members
//----------------------------------------------------------------------------------------------------------------
private:

	//data storage
	std::vector<dataEntry*> data;
	int nInputs;
	int nTargets;

	//current data set
	trainingDataSet tSet;

	//data set creation approach and total number of dataSets
	int creationApproach;
	int numTrainingSets;
	int trainingDataEndIndex;

	
//public methods
//----------------------------------------------------------------------------------------------------------------
public:

	dataReader(): creationApproach(NONE), numTrainingSets(-1) {}
	~dataReader();
	
	bool loadDataFile( const char* filename, int nI, int nT );
	void setCreationApproach();
	int getNumTrainingSets();
	
	trainingDataSet* getTrainingDataSet();
	std::vector<dataEntry*>& getAllDataEntries();

//private methods
//----------------------------------------------------------------------------------------------------------------
private:
	
	void createStaticDataSet();
	void processLine( std::string &line );
    size_t strcpy_s(char *d, size_t n, char const *s);	
};