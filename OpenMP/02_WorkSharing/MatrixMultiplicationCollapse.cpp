// Same but with collapes(n)

#include <stdio.h>
#include <iostream>
#include <omp.h>

const int dim = 5;

void printMatrix(int matrix[dim][dim]);

using namespace std;

    int main(void){

    // allocazione statica delle 3 matrici DIM x DIM
    int a[dim][dim];
    int b[dim][dim];
    int c[dim][dim];

    // inizializzazione delle matrici in parallelo
    #pragma omp parallel shared (a, b, c) num_threads(4)

    #pragma omp for schedule(static) collapse(2)
    for(int i = 0; i<dim; i++){
        for(int j = 0; j<dim; j++){
            a[i][j] = rand() % 3 + 1;
            b[i][j] = rand() % 3 + 1;
            c[i][j] = 0;
        }
    }

    printf("A:\n");
    printMatrix(a);
    printf("B:\n");
    printMatrix(b);
    printf("C:\n");
    printMatrix(c);

    // moltiplicazione in paralallelo
    #pragma omp parallel shared (a, b, c) num_threads(4)

    #pragma omp for schedule(static) collapse(3)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    printf("With collapse directiveC=A*B:\n");
    printMatrix(c);
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


