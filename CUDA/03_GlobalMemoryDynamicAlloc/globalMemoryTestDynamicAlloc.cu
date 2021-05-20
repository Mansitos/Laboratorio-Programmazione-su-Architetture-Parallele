#include <stdlib.h>
#include <stdio.h>

using namespace std;

// func declarations
__global__ void mallocTest();


// func implementations
__global__ void mallocTest(){
    size_t size = 123;
    char* ptr = (char*)malloc(size);    // allocazione DINAMIC perché malloc viene chiamata all'interno di un kernel: memoria allocata sullo heap della global memory
    if(ptr==NULL){
        printf("Malloc Error!\n");
    }else{
        memset(ptr,0,size);
        printf("Thread %d got pointer: %p\n", threadIdx.x,ptr);
        free(ptr);
    }
}

// main func
int main(){
    // Set heap size = 128 megabytes.
    // NB: this must be done before kernel launches.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<1,5>>>();  // 1 blocco da 5 threads
    cudaDeviceSynchronize();
    return 0;
}