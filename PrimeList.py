#!/usr/bin/python

import sys
import math
import time

DEBUG = 0

primeList = list()

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
    # 10^5:    0.012s           :      9,592
    # 10^6:    0.27s  -> 0.063s :     78,498
    # 10^7:    7.02s  -> 1.29s  :    664,579
    # 10^8:    3m10s  -> 27.3s  :  5,761,455
    # 10^9:           -> 11m4s  : 50,847,534
    print "Computing Primes from 1 to %d" % topN

    start_time = time.time()  #grab start_time
    serialPrime(topN)
    end_time = time.time()  #gettime();  // System.nanoTime();  // grab end_time
    delta_time = end_time - start_time
    print "Execution time: %f seconds" % delta_time

def serialPrime(topN):
    log_x = math.log(topN, 2)
    pi_x = 1.01 * topN / (log_x - 1.0)
    primeSize = int(pi_x)
    if (DEBUG > 0):
        print "n=%d  estimated primeList size=%d" % (topN, primeSize)
    prime_count = 0
    if (topN >= 2):
        prime_count += 1
        primeList.append(2)
        if (DEBUG > 0):
            print 2,
    for j in range(3, topN+1, 2):
        if (isPrime(j)):
            prime_count += 1
            primeList.append(j)
            if (DEBUG > 0):
                print j,
    print "Found %d primes" %prime_count

def isPrime(n):
    isPrime = 1
    if (n < 2):
        isPrime = 0
    dbl_sqrt_n = math.sqrt(n+1.0)
    sqrt_n = int(dbl_sqrt_n)
    if (DEBUG > 0):
        print "n=%d sqrt_n=%d" % (n, sqrt_n)
    i = 0
    list_n = 0
    while (isPrime and i < len(primeList) and list_n <= sqrt_n):
        list_n = primeList[i]
        if ((n % list_n) == 0):
            isPrime = 0
        i += 1
    return isPrime

main()
