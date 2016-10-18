/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package prime1;

import java.util.ArrayList;

/**
 *
 * @author Dan
 */
public class Prime1_x32 {

    public static final int DEBUG = 0;  // global constant
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.printf("Usage: prog <exp> or <num>,  <exp> must be less than 10");
            System.exit(1);
        }
        int topN = Integer.parseInt(args[0]);
        if (topN < 10) {
            topN = (int)Math.pow(10.0, topN);
        }   
        // 10^1:   4
        // 10^2:  25
        // 10^3: 168
        // 10^4: 0.5s -> 0.3s      //   prime1 -> primeList :  1,229
        // 10^5: 48s -> 24s -> 0.12s -> 0.036s          :      9,592
        // 10^6:            -> 3.7s  -> 0.72s  -> 0.18s :     78,498
        // 10^7:            -> 1m54s -> 17s    -> 3.1s  :    664,579
        // 10^8:                     -> 7m42s  -> 1m9s  :  5,761,455
        // 10^9:                               -> 27m5s : 50,847,534
        System.out.println("Computing Primes from 1 to " + topN);
        long start_time = System.nanoTime(); //grab start_time
        serialPrime(topN);        
        long end_time = System.nanoTime();  // grab end_time
        long delta_time = end_time - start_time;
        double delta_secs = delta_time * 1e-9; // 0.000000001;
        System.out.println("Execution time: " + delta_secs + " seconds");        
    }
    static void serialPrime(int topN) {
        double log_x = Math.log(topN);
        double pi_x = 1.01 * topN / (log_x - 1.0);
        int primeSize = (int)pi_x;
        if (true || DEBUG > 0)
            System.out.printf("n=%d  estimated primeList size=%d\n", topN, primeSize);
        int prime_count = 0;
        if (topN >= 2) {
            prime_count++;
            if (DEBUG > 0)
                System.out.print(2 + " ");
        }
        for (int j = 3; j <= topN; j += 2) {
            if (isPrime(j)) {
                prime_count++;
                if (DEBUG > 0) 
                    System.out.print(j + " ");
            }
        }
        System.out.println("Found " + prime_count + " primes");
    }
    static boolean isPrime(int n) {
        boolean isPrime = true;
        if (n < 2)
            isPrime = false;
        if ((n != 2) && ((n % 2) == 0))
            isPrime = false;
        int i = 3;
        while (isPrime && (i * i) <= n) {
            if ((n % i) == 0)
                isPrime = false;
            i += 2;
        }
        return isPrime;
    }
}
