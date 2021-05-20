// file esempio sommavettore_cpu.c

#include <stdio.h>
#define N 100   // numero valori - length arrays

using namespace std;

void add( int *a, int *b, int *c ) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main( void ) {
    int a[N], b[N], c[N];
    // l’host inizializza a[] e b[]: con valori random
    for (int i=0; i<N; i++) { a[i] = -i; b[i] = i * i; }
    // effettua (host) il calcolo:
    add( a, b, c ); // c = a + b
    // stampa i risultati:
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
return 0;
}