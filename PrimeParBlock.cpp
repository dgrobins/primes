//Program: This use Pthreads for multi-threading.
//FYI: std::thread is an alternative to Pthreads.
//FYI: Also consider openMP, MPI, and PyPy implementaions
//FYI: Also consider just multiple processes, although sharing is hard, but measure time anomaly.
// By Dan Robinson, 8/24/2016

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

// global constants
int DEBUG = 0;
int NUM_THREADS = 8;
int RUN_PARALLEL = 1;  // enable multi-threads
int GRANULARITY = 1000;  // 0 means 'coarse'. full block granularity
// Note COARSE_BLOCK=0 does not work well, due to multiples
// TBD: add GRANULARITY: which controls minor_size, minor_step, major_step
//  to allow for adjusting granularity of #of items to groups together 
// before interleaving per thread.

//public class PrimeParBlock implements PrimeGlobals {
//long primeCount[NUM_THREADS];

void *parBlockPrime(void *);
int isPrime(long);

struct thread_data {
    int threadId;
    long topN;
    long *primeCount;
};

void usage() {
    printf("Usage: prog [options] <exp> or <num>,  <exp> must be less that 10\n");
    printf("  Options:\n");
    printf("  -n <threads>      number of threads to run.  default is 1.\n");
    printf("  -g <granularity>  number of tasks per thread before interleave.\n");
    printf( "                      default 0 means 'coarse' block, ie. no interleave\n");
    printf("  -p                parallelize threads (default)\n");
    printf("  -s                serialize thread tasks in one thread\n");
    exit(EXIT_FAILURE); 
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        //TBD:usage();
        printf("Usage: %s <exp> or <num>,  <exp> must be less than 10", argv[0]);
        exit(EXIT_FAILURE);
    }
    long topN = strtol(argv[1], NULL, 10);
    if (topN < 10) {
        topN = pow(10, topN);
    }
    // 10^1:                                                   4
    // 10^2:                                                  25
    // 10^3:                                                 168
    // 10^4: 0.5s -> 0.3s      //   prime1 -> primeList :  1,229
    // 10^5: 48s -> 24s -> 0.12s -> 0.036s          :      9,592
    // 10^6:            -> 3.7s  -> 0.72s  -> 0.18s :     78,498
    // 10^7:            -> 1m54s -> 17s    -> 3.1s  :    664,579
    // 10^8:                     -> 7m42s  -> 1m9s  :  5,761,455
    // 10^9:                               -> 27m5s : 50,847,534
    printf("Computing Primes from 1 to %d\n", topN);

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    //long start_time = System.nanoTime(); //grab start_time
    long primeCount[NUM_THREADS];
    long total_prime_count = 0;
    
    //create tasks and threads
    //PrimeTask pt_task0 = new PrimeTask(0,topN);
    //Thread thread0 = new Thread(pt_task0);
    //thread0.run();

    //PrimeTask[] pt_task = new PrimeTask[NUM_THREADS];
    //Thread[] pt_thread = new Thread[NUM_THREADS];
    //long primeCount[NUM_THREADS];
    //for (int i = 0; i < NUM_THREADS; i++) {
    //    pt_task[i] = new PrimeTask(i, topN, primeCount);
    //    pt_thread[i] = new Thread(pt_task[i]);
    //}
    // execute tasks or threads(parallel)
    if (RUN_PARALLEL > 0) {
        // start threads
        pthread_t thread[NUM_THREADS];
        struct thread_data td[NUM_THREADS];
        int rc;
        int i;
        for (i = 0; i < NUM_THREADS; i++) {
            //pt_thread[i].start();
            td[i].threadId = i;
            td[i].topN = topN;
            td[i].primeCount = primeCount;
            rc = pthread_create(&thread[i], NULL, 
                                parBlockPrime, (void *)&td[i]);
            if (rc != 0) {
                printf("pthread_create() error");
                exit(EXIT_FAILURE);
            }
        }
        //wait for threads
        for (i = 0; i < NUM_THREADS; i++) {
            pthread_join(thread[i], NULL);
        }
    } else {
        for (int i = 0; i < NUM_THREADS; i++)
            1 ;  //FIXME: pt_thread[i].run();
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        total_prime_count += primeCount[i];
    }

    //long end_time = gettime();  // System.nanoTime();  // grab end_time
    gettimeofday(&end_time, NULL);
    long delta_secs = end_time.tv_sec - start_time.tv_sec;
    long delta_usecs = end_time.tv_usec - start_time.tv_usec;
    double dbl_secs = delta_secs + (double)delta_usecs / 1000000.0;
    printf("Total:   Primes=%d ExecTime=%f secs\n", total_prime_count, dbl_secs);
}

//class PrimeTask implements PrimeGlobals, Runnable {
//    int threadId;
//    long topN;
//    long[] primeCount;  // false shared array
    
//    PrimeTask(int threadId, long topN, long[] primeCount) {
//        this.threadId = threadId;
//        this.topN = topN;
//        this.primeCount = primeCount;
//    }

//    public void run() {
//        primeCount[threadId] = parBlockPrime(threadId, topN);
//        //threadId = 0;
//    }

void *parBlockPrime(void *thread_arg) {
    struct thread_data *my_data;
    my_data = (struct thread_data *) thread_arg;
    int i = my_data->threadId;
    long topN = my_data->topN;
    long *primeCount = my_data->primeCount;

    //long start_time = System.nanoTime(); //grab start_time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    long prime_count = 0;
    long minor_block_size, major_block_size;
    long block_first, block_last;
    if (GRANULARITY == 0) {
        minor_block_size = topN / NUM_THREADS;
    } else {
        minor_block_size = GRANULARITY;
    }
    major_block_size = minor_block_size * NUM_THREADS;

    //block_size = minor_block_size;
    long num_blocks = (topN + major_block_size - 1) / major_block_size;
    for (long b = 0; b < num_blocks; b++) {
        block_first = (major_block_size * b) + (minor_block_size * i) + 1;
        block_last = block_first + minor_block_size - 1;
        if (block_last > topN)
            block_last = topN;
        if (DEBUG > 0)
            printf("thr=%d blk=%d first=%d last=%d\n", i, b, block_first, block_last);
        for (long j = block_first; j <= block_last; j++) {
            if (j == 2) {
                prime_count++;
                if (DEBUG > 0)
                    printf("2 ");
            }
            if ((j % 2) == 0)
                continue;
            if (isPrime(j)) {
                prime_count++;
                if (DEBUG > 0) 
                    printf("%d ", j);
            }
        }
    }
    if (DEBUG > 0)
        printf("\n  Found %d primes\n", prime_count);
    //long end_time = System.nanoTime();  // grab end_time
    gettimeofday(&end_time, NULL);
    long delta_secs = end_time.tv_sec - start_time.tv_sec;
    long delta_usecs = end_time.tv_usec - start_time.tv_usec;
    double dbl_secs = delta_secs + (double)delta_usecs / 1000000.0;
    printf("Thread=%d Primes=%d ExecTime=%f secs\n", i, prime_count, dbl_secs);
    my_data->primeCount[i] = prime_count;  // update return value
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
