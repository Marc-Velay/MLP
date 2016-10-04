#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

using namespace std;


int main(void) {
    int posx, posy, result;

    fstream outputFile;
    outputFile.open("k-meansData.csv", ios::out);

    if( outputFile.is_open() ) {
        for( int i = 0; i < 10000; i++) {
            posx = rand()%10;
            posy = rand()%10;
            if(posx <= posy) {
                result = 1;
            } else {
                result = 0;
            }

            outputFile << posx << "," << posy << "," << result << endl;
            cout << posx << "," << posy << "," << result << endl;
        }
        outputFile.close();
    } else {
        cout << endl << "Error - Could not open file: 'k-meansData.csv' " << endl;
    }


    return 0;
}