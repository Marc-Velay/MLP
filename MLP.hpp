#include <cstdio>
#include <iostream>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <list>
#include "neurone.hpp"

using namespace std;

class MLP{
        list<Neurone*> listLayers;
        //layers[n].neurones[layerLenghts[n]]
        public:
            void train();
            void feedforward();
            string outputToClass();
            int* classToOutput(int *value);
            MLP(int nbLayers, int* layerLenghts);
                    
        private:
            void backpropagation();
            void loadData(int nbLayers, int* layerLenghts);
            void evaluate();
            void activation();
            void activationDerivate();
};
