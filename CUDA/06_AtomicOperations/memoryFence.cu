
#include <stdio.h>
#define Lenght 100000
#define NumThPerBlock 512
#define RandomMaxRange 100 // l'array verrà popolato con int random tra 0 e RandomMaxRange

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;
int NumBlocks = ceil((float)Lenght/(float)NumThPerBlock);    // casting or int-div will be performed

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
    }
}

__device__ float calculatePartialSum(const float* array, unsigned int N){
    __shared__ float result;    // per block variable

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int low = blockIdx.x * blockDim.x;
    int high = low + blockDim.x;

    if(tid<N){ // per gestire l'ultimo blocco
        if(tid >= low && tid < high){
            atomicAdd(&result,array[tid]);
        }
    }

    __syncthreads(); // altrimenti il print non ha i valori aggiornati :)

    if(threadIdx.x == 0) {
        printf("I'm block %d and my partial sum is %f\n",blockIdx.x,result);
    }

    return result; 
}

__device__ float calculateTotalSum(volatile float *result, int NumBlocks){
    float sum = 0;
    for(int i=0; i<NumBlocks; i++){
        sum+=result[i];
    } 
    return sum;
}

__global__ void sum(const float* array, unsigned int N, volatile float* result, int NumBlocks){
   
    // Ogni blocco somma una porzione dell’input:
    float partialSum = calculatePartialSum(array, N);
    //printf("block %d partial sum: %d",blockIdx.x,partialSum);

    if (threadIdx.x == 0) {
        // Il thread 0 di ogni block memoriza la somma parziale in global memory
        // Il compilatore userà un accesso diretto in memoria (no cache L1)
        // perchè result è dichiarata volatile. Cosicchè i thread dell’ultimo blocco
        // leggeranno i valori corretti computati dagli altri blocchi:
        result[blockIdx.x] = partialSum;
        // Il thread 0 effettua l’incremento di count, ma assicurandosi che sia
        // visibile solo dopo che l’aggiornamento di result sia visibile:
        __threadfence();
        unsigned int value = atomicInc(&count, gridDim.x);
        // Il thread 0 determina se appartiene all’ultimo blocco che ha
        // eseguito la atomicInc()
        isLastBlockDone = (value == (gridDim.x - 1));
    }
    // Sincronizzazione affinchè ogni thread legga il valore
    // corretto di isLastBlockDone.
    __syncthreads();
    if (isLastBlockDone) {
        // L’ultimo blocco che esegue, somma le somme parziali
        // memorizzate in result[0 .. gridDim.x-1]
        float totalSum = calculateTotalSum(result,NumBlocks);
        if (threadIdx.x == 0) {
        // Il thread 0 dell’ultimo block salva il risultato finale in global mem
        // e azzera count
        result[0] = totalSum;
        count = 0;
        }
    }
}   



/* ---- Main Program ---- */
int main (void){

    float *array;       // variabile host contenente indirizzo host
    float *dev_array;   // variabile host contenente indirizzo device
    float *device_result;
    float *dev_device_result;

    // allocazione statica della memoria su device in modalità zero-copy
    HANDLE_ERROR(cudaHostAlloc( (void**) &array, Lenght * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &device_result, NumBlocks * sizeof(float),cudaHostAllocMapped));
    
    // recupero del device pointer 
    cudaHostGetDevicePointer((void**) &dev_array,(void*)array,0);
    cudaHostGetDevicePointer((void**) &dev_device_result,(void*)device_result,0);
    
    // randomize dell'array

    int host_sum = 0;

    for(int i=0; i<Lenght; i++){
        array[i] = 1.0; //rand() % RandomMaxRange;
        host_sum+=array[i];
    }

    sum<<<NumBlocks,NumThPerBlock>>>(dev_array, Lenght, dev_device_result, NumBlocks);

    cudaDeviceSynchronize(); // wait untile device_result on host side is copied

    printf("\nHost says: Array length: %d | sum should be: %d\n",Lenght,host_sum);
    printf("Device says: Result: %f\n",device_result[0]);
        
}
