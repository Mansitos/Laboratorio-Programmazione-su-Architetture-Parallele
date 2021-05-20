// file esempio sommavettore_gpu.c
#include <stdio.h>
#define N 8192 // numero valori - length arrays
#define NumThPerBlock 256
#define NumBlocks 32

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;

__global__ void add( int *d_a, int *d_b, int *d_c ) {   // add kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) // Perchè questo test?
        d_c[tid] = d_a[tid] + d_b[tid];
}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
    }
}



int main( void ) {
    printf("Somma vettori su GPU\n");
    int a[N], b[N], c[N];       // a b c sono variabili host contenenti indirizzi host
    int *dev_a, *dev_b, *dev_c; // variabili host contenenti indirizzi device

    // allocazione STATICA della memoria sul device
    HANDLE_ERROR(cudaMalloc( (void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_c, N * sizeof(int)));

    // l’host inizializza dei valori per gli arrays a[] e b[]
    for (int i=0; i<N; i++) { a[i] = -i; b[i] = i * i; }

    // copia gli array a[] e b[] da host a device
    HANDLE_ERROR(cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // chiamata del kernel su device
    add<<<NumBlocks,NumThPerBlock>>>( dev_a, dev_b, dev_c );

    // recupera il risultato c dal vettore dev_c su device
    HANDLE_ERROR(cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // risultati
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // disalloca la memoria globale allocata sul device
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    return 0;
}