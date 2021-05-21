
#include <stdio.h>
#include <cooperative_groups.h>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

using namespace cooperative_groups;

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
    }
}

__device__ int thread_sum(int *input, int n) {
    int sum = 0;
        for(int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n / 4;
            i += blockDim.x * gridDim.x) { // accesso strided
        int4 in = ((int4*)input)[i]; // usa vector load (più efficiente)
        sum += in.x + in.y + in.z + in.w;
        }
    return sum;
}

__device__ int reduce_sum(thread_group g, int *temp, int val) {
    int lane = g.thread_rank();
   
    // ad ogni iterazione si dimezza il numero di thread attivi
    // ogni thread somma parziale temp[i] a temp[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    g.sync(); // attende che tutti i thread scrivano
    if (lane<i) val += temp[lane + i];
    g.sync(); // attende che tutti i thread leggano
    }
    return val; // nota: solo il thread 0 restituisce la somma completa
   }
   

__global__ void sum_kernel_block(int *sum, int *input, int n) {
    int my_sum = thread_sum(input, n);
    extern __shared__ int temp[];
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, my_sum);
    if (g.thread_rank() == 0) atomicAdd(sum, block_sum);
}

int main (void){

    int n = 1<<24; // lunghezza dell’array: 16M
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    int sharedBytes = blockSize * sizeof(int);

    int *sum, *data;
    int *d_sum, *d_data;

    // allocazione statica della memoria su device in modalità zero-copy
    HANDLE_ERROR(cudaHostAlloc( (void**) &data, n * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &sum,  1 * sizeof(int),cudaHostAllocMapped));
    
    // recupero del device pointer 
    cudaHostGetDevicePointer((void**) &d_data,(void*)data,0);
    cudaHostGetDevicePointer((void**) &d_sum,(void*)sum,0);
    
    // randomize dell'array data
    int host_sum = 0;

    for(int i=0; i<n; i++){
        data[i] = rand() % 100;
        host_sum+=data[i];
    }

    printf("\n -- Host computations finished\n");

    // call della reduction
    sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(d_sum, d_data, n);

    cudaDeviceSynchronize(); // wait untile device_result on host side is copied

    printf("\nHost says: Array length: %d | sum should be: %d\n",n,host_sum);
    printf("Device says: Result: %d\n",d_sum[0]);

    if(host_sum==d_sum[0]){
        printf("They match :)\n");
    }else{
        printf("Something wrong in GPU code.... :( \n");
    }
    
    return 0;
}