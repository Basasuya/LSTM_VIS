#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace std;

void trainSin() {
    freopen("srcData/sin1.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i));
    }
    printf("\nYes I Can\n");


    freopen("srcData/sin2.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i/2) + 10);
    }
    printf("\nYes I Can\n");
    
    freopen("srcData/sin3.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 20; i > 0; i -= 0.1) {
        printf("1,%.3f\n", sin(i*2) + 20);
    }
    printf("\nYes I Can\n");

    freopen("srcData/sin4.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i)/2 + 5);
    }
    printf("\nYes I Can\n");

     freopen("srcData/sin5.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i)*2 + 15);
    }
    printf("\nYes I Can\n");
}
void testSin() {
    freopen("srcData/sin6.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", cos(i));
    }
    printf("\nYes I Can\n");


    freopen("srcData/sin7.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i/2) + 20);
    }
    printf("\nYes I Can\n");
    
    freopen("srcData/sin8.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 20; i > 0; i -= 0.1) {
        printf("1,%.3f\n", sin(i*3) + 20);
    }
    printf("\nYes I Can\n");

    freopen("srcData/sin9.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i)/2 + 15);
    }
    printf("\nYes I Can\n");

     freopen("srcData/sin10.csv", "w", stdout);
    printf("Data Prediction, VIS in LSTM\n");
    for(double i = 0; i < 20; i += 0.1) {
        printf("1,%.3f\n", sin(i)*3 + 25);
    }
    printf("\nYes I Can\n");
}
int main() {
    //trainSin();
    testSin();
}

