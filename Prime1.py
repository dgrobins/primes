#!/usr/bin/python

import sys
import math
import time

DEBUG = 0;

def main():
    if len(sys.argv) != 2:
        print "Usage: %s <exp> or <num>,   <exp> must be less than 10" % sys.argv[0]
        sys.exit(1)
    else:
        topN = int(sys.argv[1])
    if topN < 10:
        topN = 10 ** topN
    # 10^1:                                4
    # 10^2:                               25
    # 10^3:                              168
    # 10^4: // prime1 -> primeList :   1,229
    # 10^5:    0.648s           :      9,592
    # 10^6:    14.7s  -> ?0.18s :     78,498
    # 10^7:    ?7.02s -> ?3.1s  :    664,579
    # 10^8:    ?3m10s -> ?1m9s? :  5,761,455
    # 10^9:           -> 27m5s  : 50,847,534
    print "Computing Primes from 1 to %d" % topN

    start_time = time.time()  #grab start_time
    start_clock = time.clock()
    serialPrime(topN)
    end_time = time.time()  #gettime();  // System.nanoTime();  // grab end_time
    end_clock = time.clock()
    delta_time = end_time - start_time
    delta_clock = end_clock - start_clock
    print "Execution time: %f seconds" % delta_time
    if DEBUG: print "Execution clock: %f seconds" % delta_clock

def serialPrime(topN):
    prime_count = 0
    if (topN >= 2):
        prime_count += 1
        if (DEBUG > 0):
            print 2,
    for j in range(3, topN+1, 2) :
        if (isPrime(j)):
            prime_count += 1
            if (DEBUG > 0):
                print j,
    if DEBUG: print
    print "Found %d primes" % prime_count

def isPrime(n):
    isPrime = 1
    if (n < 2):
        isPrime = 0
    if (n != 2) and ((n % 2) == 0):
        isPrime = 0;
    #dbl_sqrt_n = math.sqrt(n+1.0)
    #sqrt_n = int(dbl_sqrt_n)
    if (DEBUG > 1):
        print "n=%d sqrt_n=%d" % (n, sqrt_n)
    i = 3;
    while (isPrime and (i * i) <= n):
        if ((n % i) == 0):
            isPrime = 0
        i += 2
    return isPrime

main()
