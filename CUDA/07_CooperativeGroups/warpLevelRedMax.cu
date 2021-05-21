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


__inline__ __device__ int warpReducemax(int val){
    for(int offset = warpSize/2; offset>0;offset/=2){
        val = max(val,__shfl_down_sync(warpSize-1,val,offset));
    }

    return val;
}

__inline__ __device__ int blockReducemax(int val){

    static __shared__ int shared[32]; // Shared mem for 32 parial maxs
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReducemax(val);   // Each warp performs partial reduction

    if(lane == 0){
        shared[wid]=val;    // Write reduced value to shared mem
    }

    __syncthreads();    // Wait 4 all partial reds

    //read from shared mem only if the warp exsisted
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0){
        val = warpReducemax(val); // final reduce within first warp
    }

    return val;
}

__global__ void deviceReduceKernel(int *in, int* out, int N){
    int max = 0;

    //reduce multiple elements per thread
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<N; i+= blockDim.x * gridDim.x){
        max+= in[i];
    }

    max = blockReducemax(max);
    if(threadIdx.x == 0){   // primo thread del blocco
        out[blockIdx.x]=max;
    }
}

void deviceReduce(int *in, int* out, int N){
    int threads = 512;
    int blocks = min((N+threads-1)/threads,1024);

    deviceReduceKernel<<<blocks, threads>>>(in,out,N);  // primo lancio
    deviceReduceKernel<<<1,1024>>>(out,out,blocks);     // secondo lancio per somma dei risultati parziali

}


int main(void){

    int n = 1<<10; // lunghezza dell’array

    int *max, *in, *out;
    int *d_max, *d_in, *d_out;

    // allocazione statica della memoria su device in modalità zero-copy
    HANDLE_ERROR(cudaHostAlloc( (void**) &in, n * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &out, n * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &max,  1 * sizeof(int),cudaHostAllocMapped));
    
    // recupero del device pointer 
    cudaHostGetDevicePointer((void**) &d_in,(void*)in,0);
    cudaHostGetDevicePointer((void**) &d_out,(void*)out,0);
    cudaHostGetDevicePointer((void**) &d_max,(void*)max,0);
    
    // all 1 array
    for(int i=0; i<n; i++){
        in[i] = 1;
    }

    int max_val = 1000000;
    in[n/2] = max_val; // manually setting the max val


    printf("\n -- Host computations finished\n");

    // somma dell'array d_in con per-block + per-warp reduction
    // d_out struttura di supporto per salvataggio dei controlli parziali sul max
    deviceReduce(d_in,d_out,n);

    cudaDeviceSynchronize(); // wait untile device_result on host side is copied

    printf("\nHost says: Array length: %d | max should be: %d\n",n,max_val);
    printf("Device says: Result: %d\n",out[0]);

    if(max_val==out[0]){
        printf("They match :)\n");
    }else{
        printf("Something wrong in GPU code.... :( \n");
    }
    

    return 0;
}


