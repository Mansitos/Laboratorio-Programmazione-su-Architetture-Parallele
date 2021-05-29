#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>

const int dim = 64;
const int reps = 100;

void printMatrix(int matrix[dim][dim]);
void initializeMatrices(int a[dim][dim], int b[dim][dim], int c[dim][dim]);
void parallel_for_static(int a[dim][dim], int b[dim][dim], int c[dim][dim]);
void parallel_for_dynamic(int a[dim][dim], int b[dim][dim], int c[dim][dim],int chunk_size);
void parallel_for_guided(int a[dim][dim], int b[dim][dim], int c[dim][dim]);
void parallel_for_static_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim]);
void parallel_for_dynamic_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim],int chunk_size);
void parallel_for_guided_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim]);
void sequential(int a[dim][dim], int b[dim][dim], int c[dim][dim]);

using namespace std;
using namespace std::chrono;

    int main(void){

    // allocazione statica delle 3 matrici DIM x DIM
    int a[dim][dim];
    int b[dim][dim];
    int c[dim][dim];

    cout << "\nMean time (in ms) taken by function + initialization for each test (with " << reps << " tests for each type and dim="<< dim <<")" << endl;
    printf("\n");

    auto start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
    }
    auto stop = high_resolution_clock::now();
    auto init_duration = duration_cast<microseconds>(stop-start);
    cout << "Type: Initialization    Time: " << init_duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_static(a,b,c);
    }
    stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + static schedule    Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic(a,b,c,1);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:1 - default)   Time: " << duration.count()/reps <<  endl;


    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic(a,b,c,256);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:256)   Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic(a,b,c,1024);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:1024)   Time: " << duration.count()/reps <<  endl;


    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    int chunk_size = (dim*dim)/8;
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic(a,b,c,chunk_size);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:(dim*dim)/num_threads)   Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_guided(a,b,c);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + guided    Time: " << duration.count()/reps <<  endl;


    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_static_collapse(a,b,c);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + static schedule + collapse   Time: " << duration.count()/reps <<  endl;


    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic_collapse(a,b,c,256);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:256) + collapse   Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic_collapse(a,b,c,1024);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:1024) + collapse   Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    chunk_size = (dim*dim)/8;
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_dynamic_collapse(a,b,c,chunk_size);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + dynamic schedule (chunk size:(dim*dim)/num_threads) + collapse   Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        parallel_for_guided_collapse(a,b,c);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: for + guided schedule + collapse    Time: " << duration.count()/reps <<  endl;

    start = high_resolution_clock::now();
    // inizializzazione delle matrici in parallelo
    for(int i = 0; i<reps; i++){
        initializeMatrices(a,b,c);
        sequential(a,b,c);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop-start-init_duration);
    cout << "Type: sequential    Time: " << duration.count()/reps <<  endl;

} // main end


void printMatrix(int matrix[dim][dim]){
    for(int i = 0; i<dim; i++){
        for(int j = 0; j<dim; j++){
            printf("[%d]",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void parallel_for_static(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    // moltiplicazione in paralallelo
    
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(static)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void parallel_for_dynamic(int a[dim][dim], int b[dim][dim], int c[dim][dim],int chunk_size){
    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(dynamic,chunk_size)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

}

void parallel_for_guided(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(guided)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void parallel_for_static_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(static) collapse(3)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void parallel_for_dynamic_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim],int chunk_size){
    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(dynamic,chunk_size) collapse(3)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void parallel_for_guided_collapse(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(guided) collapse(3)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void initializeMatrices(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    // inizializzazione delle matrici in parallelo
    #pragma omp parallel shared (a, b, c) num_threads(8)

    #pragma omp for schedule(static)
    for(int i = 0; i<dim; i++){
        for(int j = 0; j<dim; j++){
            a[i][j] = rand() % 3 + 1;
            b[i][j] = rand() % 3 + 1;
            c[i][j] = 0;
        }
    }
}


void sequential(int a[dim][dim], int b[dim][dim], int c[dim][dim]){
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

