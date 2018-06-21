#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace std;


int main() {
    freopen("Basasuya.csv", "w", stdout);

    printf("Data Prediction, VIS in LSTM\n");

    for(double i = 0; i < 10; i += 0.1) {
        printf("1,%.3f\n", sin(i));
    }

    printf("\nYes I Can\n");


    freopen("Basasuya2.csv", "w", stdout);

    printf("Data Prediction, VIS in LSTM\n");

    for(double i = 0; i < 10; i += 0.1) {
        printf("1,%.3f\n", i);
    }

    printf("\nYes I Can\n");

    freopen("Basasuya3.csv", "w", stdout);

    printf("Data Prediction, VIS in LSTM\n");

    for(double i = 10; i > 0; i -= 0.1) {
        printf("1,%.3f\n", i);
    }

    printf("\nYes I Can\n");
}

