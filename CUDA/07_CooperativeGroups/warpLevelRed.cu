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


__inline__ __device__ int warpReduceSum(int val){
    for(int offset = warpSize/2; offset>0;offset/=2){
        val+= __shfl_down_sync(warpSize-1,val,offset);
    }

    return val;
}

__inline__ __device__ int blockReduceSum(int val){

    static __shared__ int shared[32]; // Shared mem for 32 parial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);   // Each warp performs partial reduction

    if(lane == 0){
        shared[wid]=val;    // Write reduced value to shared mem
    }

    __syncthreads();    // Wait 4 all partial reds

    //read from shared mem only if the warp exsisted
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0){
        val = warpReduceSum(val); // final reduce within first warp
    }

    return val;
}

__global__ void deviceReduceKernel(int *in, int* out, int N){
    int sum = 0;
    //reduce multiple elements per thread
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<N; i+= blockDim.x * gridDim.x){
        sum+= in[i];
    }

    sum = blockReduceSum(sum);
    if(threadIdx.x == 0){
        out[blockIdx.x]=sum;
    }
}

void deviceReduce(int *in, int* out, int N){
    int threads = 512;
    int blocks = min((N+threads-1)/threads,1024);

    deviceReduceKernel<<<blocks, threads>>>(in,out,N);
    deviceReduceKernel<<<1,1024>>>(out,out,blocks);
}


int main(void){

    int n = 1<<10; // lunghezza dell’array

    int *sum, *in, *out;
    int *d_sum, *d_in, *d_out;

    // allocazione statica della memoria su device in modalità zero-copy
    HANDLE_ERROR(cudaHostAlloc( (void**) &in, n * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &out, n * sizeof(int),cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc( (void**) &sum,  1 * sizeof(int),cudaHostAllocMapped));
    
    // recupero del device pointer 
    cudaHostGetDevicePointer((void**) &d_in,(void*)in,0);
    cudaHostGetDevicePointer((void**) &d_out,(void*)out,0);
    cudaHostGetDevicePointer((void**) &d_sum,(void*)sum,0);
    
    // randomize dell'array in
    int host_sum = 0;

    for(int i=0; i<n; i++){
        in[i] = 1; //rand() % 100;
        host_sum+=in[i];
    }

    printf("\n -- Host computations finished\n");

    deviceReduce(d_in,d_out,n);

    cudaDeviceSynchronize(); // wait untile device_result on host side is copied

    printf("\nHost says: Array length: %d | sum should be: %d\n",n,host_sum);
    printf("Device says: Result: %d\n",out[0]);

    if(host_sum==out[0]){
        printf("They match :)\n");
    }else{
        printf("Something wrong in GPU code.... :( \n");
    }
    

    return 0;
}


