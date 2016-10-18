#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <vector>
//import java.util.ArrayList;

void serialPrime(long);
int isPrime(long);

int DEBUG = 0;  // global constant

std::vector<long> primeList;    

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <exp> or <num>,  <exp> must be less than 10", argv[0]);
        exit(EXIT_FAILURE);
    }
    long topN = strtol(argv[1], NULL, 10);
    if (topN < 10) {
        topN = pow(10, topN);
    }
    // 10^1:                                4
    // 10^2:                               25
    // 10^3:                              168
    // 10^4: // prime1 -> primeList :   1,229
    // 10^5:    0.012s           :      9,592
    // 10^6:    0.27s  -> 0.063s :     78,498
    // 10^7:    7.02s  -> 1.29s  :    664,579
    // 10^8:    3m10s  -> 27.3s  :  5,761,455
    // 10^9:           -> 11m4s  : 50,847,534
    printf("Computing Primes from 1 to %d\n", topN);
    
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    serialPrime(topN);        
    
    gettimeofday(&end_time, NULL);
    long delta_secs = end_time.tv_sec - start_time.tv_sec;
    long delta_usecs = end_time.tv_usec - start_time.tv_usec;
    double dbl_secs = delta_secs + (double)delta_usecs / 1000000.0;
    printf("Execution time: %f seconds\n", dbl_secs);  
}

void serialPrime(long topN) {
    double log_x = log2(topN);
    double pi_x = 1.01 * topN / (log_x - 1.0);
    int primeSize = (int)pi_x;
    if (1 || DEBUG > 0)
        printf("n=%d  estimated primeList size=%d\n", topN, primeSize);
    primeList.reserve(primeSize);
    long prime_count = 0;
    if (topN >= 2) {
        prime_count++;
        primeList.push_back(2);
        if (DEBUG > 0)
            printf("%d ", 2);
    }
    for (long j = 3; j <= topN; j += 2) {
        if (isPrime(j)) {
            prime_count++;
            primeList.push_back(j);
            if (DEBUG > 0) 
                printf("%d ", j);
        }
    }
    printf("Found %d primes\n", prime_count);
}

int isPrime(long n) {
    int isPrime = 1;
    if (n < 2)
        isPrime = 0;
    double dbl_sqrt_n = sqrt(n+1.0);
    long sqrt_n = (long)dbl_sqrt_n;
    if (DEBUG > 0) 
        printf("n=%d sqrt_n=%d\n", n, sqrt_n);
    int i = 0;
    long list_n = 0;
    while (isPrime && i < primeList.size() && list_n <= sqrt_n) {
        list_n = primeList[i];
        if ((n % list_n) == 0)
            isPrime = 0;
        i++;
    }
    return isPrime;
}
