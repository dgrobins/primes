/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package prime;

import java.util.ArrayList;
import java.util.*;


/**
 *  Parallel Implementation of Prime using *(coarse or fine) block assignment
 * @author Dan
 */

interface PrimeGlobals {
    // global constants
    public static final int DEBUG = 0;
    public static final int NUM_THREADS = 8;
    public static final int RUN_PARALLEL = 1;  // enable multi-threads
    public static final int GRANULARITY = 1000;  // 0 means 'coarse'. full block granularity  
    // Note COARSE_BLOCK=0 does not work well, due to multiples
    // TBD: add GRANULARITY: which controls minor_size, minor_step, major_step
    //  to allow for adjusting granularity of #of items to groups together 
    // before interleaving per thread.
}
public class PrimeParBlock implements PrimeGlobals {
    static long[] primeCount;
    /**
     * @param args the command line arguments
     */
    static void usage() {
        System.out.printf("Usage: prog [options] <exp> or <num>,  <exp> must be less that 10\n" +
                          "  Options:\n" +
                          "  -n <threads>      number of threads to run.  default is 1.\n" +
                          "  -g <granularity>  number of tasks per thread before interleave.\n" +
                          "                      default 0 means 'coarse' block, ie. no interleave\n" +
                          "  -p                parallelize threads (default)\n" +
                          "  -s                serialize thread tasks in one thread\n");
        System.exit(1); 
    }
    public static void main(String[] args) {
        if (args.length != 1) {
            //TBD:usage();
            System.out.printf("Usage: prog <exp> or <num>,  <exp> must be less than 10");
            System.exit(1);
        }
        long topN = Long.parseLong(args[0]);
        if (topN < 10) {
            topN = (long)Math.pow(10.0, topN);
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
        System.out.println("Computing Primes from 1 to " + topN);
        long start_time = System.nanoTime(); //grab start_time
        long total_prime_count = 0;
        
        //create tasks and threads
        //PrimeTask pt_task0 = new PrimeTask(0,topN);
        //tba:PrimeTask pt_task1 = new PrimeTask(1,100);
        //Thread thread0 = new Thread(pt_task0);
        //tba:Thread thread1 = new Thread(pt_task1);
        //thread0.run();
        //tba:thread1.run();
        
        PrimeTask[] pt_task = new PrimeTask[NUM_THREADS];
        Thread[] pt_thread = new Thread[NUM_THREADS];
        primeCount = new long[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++) {
            pt_task[i] = new PrimeTask(i, topN, primeCount);
            //if (RUN_PARALLEL > 0)
            pt_thread[i] = new Thread(pt_task[i]);
        }
        // execute tasks or threads(parallel)
        if (RUN_PARALLEL > 0) {
            // start threads
            for (int i = 0; i < NUM_THREADS; i++)
                pt_thread[i].start();
            //wait for threads
            try {
                for (int i = 0; i < NUM_THREADS; i++)
                    pt_thread[i].join();
            }
            catch (InterruptedException ex){
            }
        } else {
            //for (int i = 0; i < NUM_THREADS; i++)
            for (int i = 0; i < NUM_THREADS; i++)
                pt_thread[i].run();
        }
        
        for (int i = 0; i < NUM_THREADS; i++) {
            total_prime_count += primeCount[i];
        }
        long end_time = System.nanoTime();  // grab end_time
        long delta_time = end_time - start_time;
        double delta_secs = delta_time * 1e-9; // 0.000000001;
        System.out.printf("Total:   Primes=%d ExecTime=%f secs\n", 
                total_prime_count, delta_secs);
    }
}

class PrimeTask implements PrimeGlobals, Runnable {
    int threadId;
    long topN;
    long[] primeCount;  // false shared array
    
    PrimeTask(int threadId, long topN, long[] primeCount) {
        this.threadId = threadId;
        this.topN = topN;
        this.primeCount = primeCount;
    }

    public void run() {
        primeCount[threadId] = parBlockPrime(threadId, topN);
        //threadId = 0;
    }

    long parBlockPrime(int threadId, long topN) {
        int i = threadId;
        long start_time = System.nanoTime(); //grab start_time
        //tbr:assert (RUN_PARALLEL > 0);
        //tbr:if (RUN_PARALLEL > 0)
        //tbr:    i = 0; //i = ThreadID.get();  // thread IDs are in {0..NUM_THREADS-1}
        double log_x = Math.log(topN);
        double pi_x = 1.01 * topN / (log_x - 1.0);
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
                System.out.printf("thr=%d blk=%d first=%d last=%d\n",
                        i, b, block_first, block_last);
            for (long j = block_first; j <= block_last; j++) {
                if (j == 2) {
                    prime_count++;
                    if (DEBUG > 0)
                        System.out.print(2 + " ");
                }
                if ((j % 2) == 0)
                    continue;
                if (isPrime(j)) {
                    prime_count++;
                    if (DEBUG > 0) 
                        System.out.print(j + " ");
                }
            }
        }
        if (DEBUG > 0)
            System.out.printf("\n  Found %d primes\n", prime_count);
        long end_time = System.nanoTime();  // grab end_time
        long delta_time = end_time - start_time;
        double delta_secs = delta_time * 1e-9; // 0.000000001;
        System.out.printf("Thread=%d Primes=%d ExecTime=%f secs\n", i, prime_count, delta_secs);        
        return prime_count;
    }

    boolean isPrime(long n) {
        boolean isPrime = true;
        if (n < 2)
            isPrime = false;
        if ((n != 2) && ((n % 2) == 0))
            isPrime = false;
        long i = 3;
        while (isPrime && (i * i) <= n) {
            if ((n % i) == 0)
                isPrime = false;
            i += 2;
        }
        return isPrime;
    }
}
