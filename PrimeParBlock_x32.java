/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prime;

import java.util.ArrayList;

/**
 *  Parallel Implementaion of Prime using *(coarse or fine) block assignment
 * @author Dan
 */
public class PrimeParBlock_x32 {

    // global constant
    public static final int DEBUG = 0;
    public static final int NUM_THREADS = 1;  
    public static final int RUN_PARALLEL = 0;  // enable multi-threads
    public static final int GRANULARITY = 0;  // 0 means 'coarse'. full block granularity  
    // Note COARSE_BLOCK=0 does not work well, due to multiples
    // TBD: add GRANULARITY: which controls minor_size, minor_step, major_step
    //  to allow for adjusting granularity of #of items to groups together 
    // before interleaving per thread.
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        //primePrint();
        //int topN = (int)Math.pow(10,9);
        int topN = 1000000;
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
        int total_prime_count = 0;
        for (int i = 0; i < NUM_THREADS; i ++) {
            total_prime_count += parBlockPrime(topN, i);
        }
        long end_time = System.nanoTime();  // grab end_time
        long delta_time = end_time - start_time;
        double delta_secs = delta_time * 1e-9; // 0.000000001;
        System.out.printf("Total:   Primes=%d ExecTime=%f secs\n", 
                total_prime_count, delta_secs);
    }
    static int parBlockPrime(int topN, int i) {
        long start_time = System.nanoTime(); //grab start_time
        assert (RUN_PARALLEL > 0);
        if (RUN_PARALLEL > 0)
            i = 0; //i = ThreadID.get();  // thread IDs are in {0..NUM_THREADS-1}
        double log_x = Math.log(topN);
        double pi_x = 1.01 * topN / (log_x - 1.0);
        int prime_count = 0;
        int minor_block_size, major_block_size;
        int block_first, block_last;
        if (GRANULARITY == 0) {
            minor_block_size = topN / NUM_THREADS;
        } else {
            minor_block_size = GRANULARITY;
        }
        major_block_size = minor_block_size * NUM_THREADS;

        //block_size = minor_block_size;
        int num_blocks = (topN + major_block_size - 1) / major_block_size;
        for (int b = 0; b < num_blocks; b++) {
            block_first = (major_block_size * b) + (minor_block_size * i) + 1;
            block_last = block_first + minor_block_size - 1;
            if (block_last > topN)
                block_last = topN;
            if (DEBUG > 0)
                System.out.printf("thr=%d blk=%d first=%d last=%d\n",
                        i, b, block_first, block_last);
            for (int j = block_first; j <= block_last; j++) {
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
    static boolean isPrime(int n) {
        boolean isPrime = true;
        if (n < 2)
            isPrime = false;
        int sqrt_n = (int)Math.sqrt(n + 1);
        if (DEBUG > 1) 
            System.out.printf("n=%d sqrt_n=%d\n", n, sqrt_n);
        int i = 2;
        while (isPrime && i <= sqrt_n) {
            if ((n % i) == 0)
                isPrime = false;
            i++;
        }
        return isPrime;
    }
}
