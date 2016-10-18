/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package primeList;

import java.util.ArrayList;

/**
 *
 * @author Dan
 */
public class PrimeList {

    public static final int DEBUG = 0;  // global constant
    
    static ArrayList<Long> primeList;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.printf("Usage: prog <exp> or <num>,  <exp> must be less than 10");
            System.exit(1);
        }
        long topN = Long.parseLong(args[0]);
        if (topN < 10) {
            topN = (long)Math.pow(10.0, topN);
        }   
        // 10^4: 0.5s -> 0.3s
        // 10^5: 48s -> 24s -> 0.12s
        // 10^6:            -> 3.7s -> 0.75s           :     78,498
        // 10^7:            -> 1m54s -> 17s -> 3.1s    :    664,579
        // 10^8:                     -> 7m42s -> 1m9s  :  5,761,455
        // 10^9:                              -> 27m5s : 50,847,534
        System.out.println("Computing Primes from 1 to " + topN);
        long start_time = System.nanoTime(); //grab start_time
        serialPrime(topN);        
        long end_time = System.nanoTime();  // grab end_time
        long delta_time = end_time - start_time;
        double delta_secs = delta_time * 1e-9; // 0.000000001;
        System.out.println("Execution time: " + delta_secs + " seconds");        
    }
    static void serialPrime(long topN) {
        double log_x = Math.log(topN);
        double pi_x = 1.01 * topN / (log_x - 1.0);
        int primeSize = (int)pi_x;
        if (true || DEBUG > 0)
            System.out.printf("n=%d  estimated primeList size=%d\n", topN, primeSize);
        primeList = new ArrayList<Long>(primeSize);
        long prime_count = 0;
        if (topN >= 2) {
            prime_count++;
            primeList.add(2L);
            if (DEBUG > 0)
                System.out.print(2 + " ");
        }
        for (long j = 3; j <= topN; j += 2) {
            if (isPrime(j)) {
                prime_count++;
                primeList.add(j);
                if (DEBUG > 0) 
                    System.out.print(j + " ");
                //if ((prime_count % 100) == 0)
                //    System.out.println();
            }
        }
        System.out.println("Found " + prime_count + " primes");
    }
    static boolean isPrime(long n) {
        boolean isPrime = true;
        if (n < 2)
            isPrime = false;
        long sqrt_n = (long)Math.sqrt(n + 1);
        if (DEBUG > 0) 
            System.out.printf("n=%d sqrt_n=%d\n", n, sqrt_n);
        int i = 0;
        long list_n = 0;
        while (isPrime && i < primeList.size() && list_n <= sqrt_n) {
            list_n = primeList.get(i);
            //for (long i = 2; i < sqrt_n; i++) {
            if ((n % list_n) == 0)
                isPrime = false;
            i++;
        }
        return isPrime;
    }
/*    static void primePrint() {
        int i = ThreadID.get();  // thread IS are in {0..9}
        int block = power(10,9);
        for (int j = (i * block) + 1; j <= (i + 1) * block; j++) {
                if (isPrime(j))
                    print(j);
        }
    }
*/
}
