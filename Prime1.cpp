#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <string>

void serialPrime(long);
int isPrime(long);

int DEBUG = 0;

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
    // 10^6:    0.27s  -> ?0.18s :     78,498
    // 10^7:    7.02s  -> ?3.1s  :    664,579
    // 10^8:    3m10s  -> ?1m9s? :  5,761,455
    // 10^9:                               -> 27m5s : 50,847,534
	printf("Computing Primes from 1 to %d\n", topN);

	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);
    //long start_time = gettime();  // System.nanoTime(); //grab start_time
    serialPrime(topN);        
    //long end_time = gettime();  // System.nanoTime();  // grab end_time
    gettimeofday(&end_time, NULL);
    long delta_secs = end_time.tv_sec - start_time.tv_sec;
    long delta_usecs = end_time.tv_usec - start_time.tv_usec;
    //double delta_secs = delta_time * 1e-9; // 0.000000001;
    double dbl_secs = delta_secs + (double)delta_usecs / 1000000.0;
    printf("Execution time: %f seconds\n", dbl_secs);  
}

void serialPrime(long topN) {
    //double log_x = Math.log(topN);
    //double pi_x = 1.01 * topN / (log_x - 1.0);
    //int primeSize = (int)pi_x;
    //if (true || DEBUG > 0)
    //    System.out.printf("n=%d  estimated primeList size=%d\n", topN, primeSize);
    long prime_count = 0;
    if (topN >= 2) {
        prime_count++;
        if (DEBUG > 0)
            printf("%d ", 2);
    }
    for (long j = 3; j <= topN; j += 2) {
        if (isPrime(j)) {
            prime_count++;
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
    if ((n != 2) && ((n % 2) == 0))
        isPrime = 0;
    long i = 3;
    while (isPrime && (i * i) <= n) {
        if ((n % i) == 0)
            isPrime = 0;
        i += 2;
    }
    return isPrime;
}
