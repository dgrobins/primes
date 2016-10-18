/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package primeDynExec;

import java.util.ArrayList;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 *  Parallel Implementation of Prime using *(coarse or fine) block assignment
 * @author Dan
 */

interface PrimeGlobals {
    // global constants
    public static final int DEBUG = 0;
    public static final int NUM_THREADS = 8;
    public static final int RUN_PARALLEL = 1;  // enable multi-threads
    public static final int COUNTER = 1;  // 0:unprotected counter, 1:atomic access counter, 2:synchronized counter
    public static final int EXECUTOR = 0;  // use thread 'executor'
    public static final int GRANULARITY = 100;  // 0 means 'coarse'. full block granularity  
    // Note COARSE_BLOCK=0 does not work well, due to multiples
    // TBD: add GRANULARITY: which controls minor_size, minor_step, major_step
    //  to allow for adjusting granularity of #of items to groups together 
    // before interleaving per thread.
}
public class PrimeParDynExec implements PrimeGlobals {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        //primePrint();
        //int topN = (int)Math.pow(10,9);
        long topN = 10000000L;
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
        //tbr:Thread[] pt_thread = new Thread[NUM_THREADS];
        long[] primeCount = new long[NUM_THREADS];
        if (COUNTER == 0)
            Counter counter = new Counter();
        elif (COUNTER == 1)
            Counter syncCounter = new AtomicCounter();
        elif (COUNTER == 2)
            Counter counter = new SyncCounter();
        else
            system.exit(1)
        //AtomicLong atomicCounter = new AtomicLong(0);
        
        for (int i = 0; i < NUM_THREADS; i++) {
            pt_task[i] = new PrimeTask(i, topN, primeCount, counter);
            //if (ATOMIC == 0)
            //    pt_task[i] = new PrimeTask(i, topN, primeCount, synchCounter);
            //else
            //    pt_task[i] = new PrimeTask(i, topN, primeCount, atomicCounter);
            //pt_thread[i] = new Thread(pt_task[i]);
        }

        if (RUN_PARALLEL == 0) {
            // serialize thread tasks
            for (int i = 0; i < NUM_THREADS; i++)
                pt_thread[i].run();                
        } 
        elif (EXECUTOR == 0) {
            // execute tasks or threads(parallel), manually
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
        }
        elif (EXECUTOR == 1) {
            // execute tasks or threads(parallel), using Executors class 
            //ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            ExecutorService executor = Executors.newCachedThreadPool();
            // start threads
            for (int i = 0; i < NUM_THREADS; i++)
                executor.execute(pt_task[i]);
            executor.shutdown();
            //wait for threads
            while (!executor.isTerminated()) {
            }
        }
        else {
            system.exit(1);
        }
        // compute sum primeCount[]
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

class Counter {
    long value = 0;
    Counter() { value = 0; }
    long getAndAdd(long add) {
        long old_value = value;
        value += add;
        return old_value;
    }
}

class SyncCounter {
    long value;
    Counter() {
        value = 0;
    }
    synchronized long getAndAdd(long add) {
        long old_value = value;
        value += add;
        return old_value;
    }
}

class PrimeTask implements PrimeGlobals, Runnable {
    int threadId;
    long topN;
    long[] primeCount;  // false shared array
    Counter synchCounter;
    AtomicLong atomicCounter;
    
    PrimeTask(int threadId, long topN, long[] primeCount, Counter synchCounter) {
        this.threadId = threadId;
        this.topN = topN;
        this.primeCount = primeCount;
        this.synchCounter = synchCounter;
    }
    PrimeTask(int threadId, long topN, long[] primeCount, AtomicLong atomicCounter) {
        this.threadId = threadId;
        this.topN = topN;
        this.primeCount = primeCount;
        this.atomicCounter = atomicCounter;
    }


    public void run() {
        primeCount[threadId] = parBlockPrime(threadId, topN);
        //threadId = 0;
    }

    //static long parBlockPrime(int threadId, long topN) {
    long parBlockPrime(int threadId, long topN) {
        int i = threadId;
        long start_time = System.nanoTime(); //grab start_time
        //tbr:assert (RUN_PARALLEL > 0);
        //tbr:if (RUN_PARALLEL > 0)
        //tbr:    i = 0; //i = ThreadID.get();  // thread IDs are in {0..NUM_THREADS-1}
        double log_x = Math.log(topN);
        double pi_x = 1.01 * topN / (log_x - 1.0);
        long prime_count = 0;
        long minor_block_size;
        long block_first, block_last;
        if (GRANULARITY == 0) {
            minor_block_size = topN / NUM_THREADS;
        } else {
            minor_block_size = GRANULARITY;
        }

        //block_size = minor_block_size;
        block_first = 0;
        int b = 0;
        while (block_first < topN) {
            if (ATOMIC == 0)
                block_first = synchCounter.getAndAdd(minor_block_size);
            else
                block_first = atomicCounter.getAndAdd(minor_block_size);
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
            b++;
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
        long sqrt_n = (long)Math.sqrt(n + 1);
        if (DEBUG > 1) 
            System.out.printf("n=%d sqrt_n=%d\n", n, sqrt_n);
        long i = 2;
        while (isPrime && i <= sqrt_n) {
            if ((n % i) == 0)
                isPrime = false;
            i++;
        }
        return isPrime;
    }
}
