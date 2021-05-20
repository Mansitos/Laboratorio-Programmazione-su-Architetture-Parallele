// file di esempio/test compilazione

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void my_kernel( void ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf( "Sono il thread %d! block %d!\n", tid , blockIdx.x);
}

int main( void ) {  
    my_kernel<<<2,32>>>();  // 2 blocchi da 32 thread ciascuno
    printf( "Hello, World!\n" );
    cout << "Inserire qualsiasi carattere per interrompere il processo\n";
    string x;
    cin >> x;
    return 0;
}