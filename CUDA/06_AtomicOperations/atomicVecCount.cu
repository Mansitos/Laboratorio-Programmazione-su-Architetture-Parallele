/*
Scrivere un programma che dato un array di N intery vec[] e un valore intero x conta quanti sonogli elementi di vec[] uguale a x.

- host alloca vec[] e lo inizializza (random)
- B blocks da 256threads (gestire i casi in cui B non è multiplo di N)
- ogni thread accede ad un elementi dell'array, se è uguale ad x incrementa un counter (in modo atomico!) posizionato in global memory
- host recupera il risultato e lo stampa + check correttezza
*/

#include <stdio.h>
#define N 16390
#define NumThPerBlock 512
#define RandomMaxRange 100 // l'array verrà popolato con int random tra 0 e RandomMaxRange
#define value 50 // il valore da cercare (dev'essere int e compreso >0 <RandomMaxRange)

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;
int NumBlocks = ceil((float)N/(float)NumThPerBlock);    // casting or int-div will be performed

__global__ void counter(int *vec_to_check, int value_to_count, int *device_counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){ // Per i thread dell'ultimo blocco nei casi "non-multipli"
        if(vec_to_check[tid] == value_to_count){
            device_counter[1]++;    // without atomic add (for check differences)
            atomicAdd(&device_counter[0],1);    // with atomic add
        }
    }else{
        printf("I'm the thread %d and now I'll sleep cause I'm useless for the computation :)\n",tid);
    }
}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
    }
}

/* ---- Main Program ---- */
int main (void){
    printf("N: %d | Blocks: %d with %d threds each",N,NumBlocks,NumThPerBlock);

    int *vec;       // variabile host contenente indirizzo host
    int *dev_vec;   // variabile host contenente indirizzo device
    int *dev_device_counter;    // variabile host contenente indirizzo host
    int *device_counter;        // variabile host contenente indirizzo device

    // allocazione statica della memoria su device in modalità zero-copy
    HANDLE_ERROR(cudaHostAlloc( (void**) &vec, N * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &device_counter, 2*sizeof(int),cudaHostAllocMapped));

    // recupero del device pointer 
    cudaHostGetDevicePointer((void**) &dev_vec,(void*)vec,0);
    cudaHostGetDevicePointer((void**) &dev_device_counter,(void*)device_counter,0);

    // randomize dell'array
    for(int i=0; i<N; i++){
        vec[i] = rand() % RandomMaxRange;
    }

    //kernel call
    counter<<<NumBlocks,NumThPerBlock>>>(dev_vec,value,dev_device_counter);

    cudaDeviceSynchronize(); // wait untile device_counter on host side is copied
    
    // execute check on host-side
    int host_counter = 0;
    for(int i=0; i<N; i++){
        if(vec[i] == value){
            host_counter++;
        }
    }

    // check results
    printf("\nOccurences of %d found:",value);
    printf("\ndev counter with atomic-add: %d",device_counter[0]);
    printf("\ndev counter without atomic-add: %d",device_counter[1]);
    printf("\nhost counter: %d", host_counter);

    return 0;


}