#include <stdio.h>
#include <iostream>
#include <omp.h>

using namespace std;

int main(void){
    printf("\nDirettiva parallel su %d threads con ciclo for da 5, quindi %d stampe totali.\n",omp_get_num_procs(),omp_get_num_procs()*5);

    #pragma omp parallel
    {
        for(int i = 0; i<5; i++){
            printf("Codice parallelo eseguito da %d thread\n", omp_get_num_threads());
        }
    } // Fine blocco di codice parallelo

    printf("\nFor da 10 print ripartito tra %d threads.\n", omp_get_num_procs());

    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i<10; i++){
            printf("Codice parallelo eseguito da %d thread (io sono: %d)\n", omp_get_num_threads(),omp_get_thread_num());
        }
    } // Fine blocco di codice parallelo


} // main end

