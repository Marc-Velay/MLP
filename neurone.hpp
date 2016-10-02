#include <cstdio>
#include <stdlib.h>     /* srand, rand */

class Neurone {
    //float value = sum(layers[n-1].neurones[0-max].value*W[0-max]);
    float value;
    float w [];

    public: 
        void setValues();
        Neurone();
        void setWeight();

        private:
        void initWeight();
};