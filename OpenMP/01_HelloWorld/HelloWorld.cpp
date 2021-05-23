#include <stdio.h>
#include <iostream>
#include <omp.h>

using namespace std;

int main(void){
    #pragma omp parallel
    {
        printf("Codice parallelo eseguito da %d thread\n", omp_get_num_threads());
    } // Fine blocco di codice parallelo

} // main end