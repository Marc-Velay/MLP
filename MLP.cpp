#include "MLP.hpp"

void MLP::backpropagation() {

}

void MLP::loadData(int nbLayers, int* layerLenghts) {        
        for(int i=0; i<nbLayers;++i){
                Neurone *layer=new Neurone[layerLenghts[i]];
                for(int j=0; j<layerLenghts[i];++j){
                        Neurone neurone = Neurone();
                        layer[j]=neurone;
                }
                listLayers.push_back(layer);
        }

}


void MLP::evaluate() {
        
}

void MLP::activation() {
        
}

void MLP::activationDerivate() {
        
}

void MLP::train() {
        
}

void MLP::feedforward() {
        
}

string MLP::outputToClass() {
        
        return "output";        
}

int* MLP::classToOutput(int *value) {
        return value;
}

MLP::MLP(int nbLayers, int* layerLenghts){
        loadData(nbLayers,layerLenghts);
}



int main(void){
            printf("Hello World !\n\n");
            /* initialize random seed: */
            srand (time(NULL));
            
            int layerLenghts []= {18,20,22,24,22,20,8};
            MLP *mlp = new MLP(7,layerLenghts);
            return 0;           
}