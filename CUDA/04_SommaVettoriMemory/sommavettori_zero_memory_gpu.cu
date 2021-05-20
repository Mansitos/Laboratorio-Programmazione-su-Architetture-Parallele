// file esempio sommavettore_gpu.c
#include <stdio.h>
#define N 8192
#define NumThPerBlock 256
#define NumBlocks 32

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;
bool print = false;

__global__ void add( int *d_a, int *d_b, int *d_c ) {
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
    int *a,*b,*c;       // a b c sono variabili host contenenti indirizzi host
    int *dev_a, *dev_b, *dev_c;  // variabili host contenenti indirizzi devices

    // allocazione STATICA della memoria sul device
    HANDLE_ERROR(cudaHostAlloc( (void**) &a, N * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &b, N * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &c, N * sizeof(int),cudaHostAllocMapped));

    cudaHostGetDevicePointer((void**) &dev_a,(void*)a,0);
    cudaHostGetDevicePointer((void**) &dev_b,(void*)b,0);
    cudaHostGetDevicePointer((void**) &dev_c,(void*)c,0);

    // l’host inizializza dei valori per gli arrays a[] e b[]
    for (int i=0; i<N; i++) { a[i] = -i; b[i] = i * i; }

    // copia gli array a[] e b[] da host a device (NON PIU' NECESSARIO)
    //HANDLE_ERROR(cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<NumBlocks,NumThPerBlock>>>( dev_a, dev_b, dev_c);

    cudaDeviceSynchronize();    // altrimenti printo prima che i dati vengano copiati
    
    // recupera il risultato c dal vettore dev_c su device (NON PIU' NECESSARIO)
    // HANDLE_ERROR(cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
   
    // risultati
    if(print == true){
        for (int i=0; i<N; i++) {
            printf( "%d + %d = %d\n", a[i], b[i], c[i] );
        }
    }

    printf("Done!");

    // disalloca la memoria globale allocata sul device
    cudaFreeHost( dev_a );
    cudaFreeHost( dev_b );
    cudaFreeHost( dev_c );
    return 0;
}