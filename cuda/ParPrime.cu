#include <stdio.h>

// cuda runtime
//#include <cuda_runtime.h>

// global defines
#define DEBUG 0
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65536
#define MAPPING 3  // 1:1to1, 2:2to1, 3:6to2 mapping

	// Run stats:
    // 10^1:                                                   4
    // 10^2:                                                  25
    // 10^3:                                                 168
    // 10^4: 0.5s -> 0.3s      //   prime1 -> primeList :  1,229
    // 10^5: 48s -> 24s -> 0.12s -> 0.036s          :      9,592
    // 10^6:            -> 3.7s  -> 0.72s  -> 0.18s :     78,498
    // 10^7:            -> 1m54s -> 17s    -> 3.1s  :    664,579
    // 10^8:                     -> 7m42s  -> 1m9s  :  5,761,455
    // 10^9:                               -> 27m5s : 50,847,534
    // 10^10: >32b                                  : 455,052,511
    // 10^11:                                       : 4,118,054,813
    // 10^12:                                       : 37,607,912,018
    // 10^13:										: 346,065,536,839
    // 10^14:										: 3,204,941,750,802
    // 10^15:										: 29,844,570,422,669
    // 10^20: >64b

// test: return idx in cnt[], if <= topN
__global__ void kernel_test(char *cnt, long topN)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long n = idx;
	if (n <= topN)
		cnt[idx] = idx;
}

// 1to1 map: n = idx + 2  {idx:0..k => 2,3..k+2(=topN)}
__global__ void kernel_1to1(char *cnt, long topN)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	long n = idx + 2;
	char isPrime = 1;
	if ((n < 2) || (n > topN))
		// mark n > topN as non-prime, so don't waste time computing
		isPrime = 0;
	long i = 2;
  	while (isPrime && (i * i) <= n) {
		if ((n % i) == 0) 
			isPrime = 0;
		if (DEBUG) printf("idx=%d: n=%d i=%d isPrime=%d\n", idx, n, i, isPrime);
		i++;
	}
	if (n <= topN)
		cnt[idx] = isPrime;
	if (DEBUG) printf("idx=%d num=%d topN=%d cnt[idx]=%d\n", idx, n, topN, cnt[idx]);
}

// 2to1 map: n = 2*idx + 3  {idx:0..k => 3,5,...2+1..2k+3(=topN)
__global__ void kernel_2to1(char *cnt, long topN)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	long n = 2*idx + 3;
	char isPrime = 1;
	if ((n < 2) || (n > topN))
		// mark n > topN as non-prime, so don't waste time computing
		isPrime = 0;
	//assert n%2==0
	// only check odds
	long i = 3;
  	while (isPrime && (i * i) <= n) {
		if ((n % i) == 0)
			isPrime = 0;
		if (DEBUG) printf("idx=%d: n=%d i=%d isPrime=%d\n", idx, n, i, isPrime);
		i += 2;
	}
	if (idx < topN)
		cnt[idx] = isPrime;
	if (DEBUG) printf("idx=%d num=%d topN=%d cnt[idx]=%d\n", idx, n, topN, cnt[idx]);
}

// 6to2 map: n = 6*idx/2 + 2*(idx%2) + 5  {idx:0,1,2,3,..k => 5,7,11,13,3k+2*(k%2)+5   // x3, 5, 7, x9, 11, 13, x15
__global__ void kernel_6to2(char *cnt, long topN)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	long n = 6*(idx/2) + 2*(idx%2) + 5;
	char isPrime = 1;
	if ((n < 2) || (n > topN))
		// mark n > topN as non-prime, so don't waste time computing
		isPrime = 0;
	// only check odds
	// assert n%2==0 and n%3==0
	long i = 5;
  	while (isPrime && (i * i) <= n) {
		if ((n % i) == 0)
			isPrime = 0;
		if (DEBUG) printf("idx=%d: n=%d i=%d isPrime=%d\n", idx, n, i, isPrime);
		i += 2;
		if ((n % i) == 0)
			isPrime = 0;
		if (DEBUG) printf("idx=%d: n=%d i=%d isPrime=%d\n", idx, n+2, i, isPrime);
		i += 4;
	}
	if (idx < topN)
		cnt[idx] = isPrime;
	if (DEBUG) printf("idx=%d num=%d topN=%d cnt[idx]=%d\n", idx, n, topN, cnt[idx]);
}

int main(int argc, char* argv[])
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

    if (argc != 2 && argc != 3) {
        printf("Usage: %s <exp or num> [threadsPerBlock],  <exp> must be less than 20", argv[0]);
        exit(EXIT_FAILURE);
    }
    long long topN = strtol(argv[1], NULL, 10);
    if (topN < 20) {
        topN = pow(10, topN);
    }
    long threadsPerBlock = 0;
    if (argc == 3) {
    	threadsPerBlock = strtol(argv[2], NULL, 10);
    }
    // if threadsPerBlock=0, then use max threads per block, from CONSTANT (tbd from device Query)
    if (threadsPerBlock == 0 || threadsPerBlock > MAX_THREADS_PER_BLOCK) {
		threadsPerBlock = MAX_THREADS_PER_BLOCK;
    }

    printf("Computing Primes from 1 to %d\n", topN);

    // get starting timestamp
    //struct timeval start_time, end_time;
    //gettimeofday(&start_time, NULL);
    //struct rusage rusage_start;
    //getrusage(RUSAGE_SELF, rusage_start);
	clock_t start, end;
	double cpuTime;
	//start = clock();

    // compute totalThreads, blocksPerGrid.  threadsPerBlock(perBlock) is a parameter
	// 1to1 map: n = idx + 2  {idx:0..k => 2,3..k+2(=topN)}
	// 2to1 map: n = 2*idx + 3  {idx:0..k => 3,5,...2+1..2k+3(=topN)
	// 6to2 map: n = 6*idx/2 + 2*(idx%2) + 5  {idx:0,1,2,3,..k => 5,7,11,13,3k+2*(k%2)+5   // x3, 5, 7, x9, 11, 13, x15
    long totThreads;
    if (MAPPING == 1)
    	totThreads = topN - 2 + 1;
    else if (MAPPING == 2)
		totThreads = (topN - 3) / 2 + 1;
	else if (MAPPING == 3)
		totThreads = (topN - 5) / 3 + 1 + 1;
	else {
		printf("ERROR: illegal MAPPING=%d", MAPPING);
		exit(EXIT_FAILURE);
	}
	long blocksPerGrid = (totThreads + threadsPerBlock - 1) / threadsPerBlock;
    long num_bytes = totThreads * sizeof(char);

    char *d_cnt = 0, *h_cnt = 0;
    h_cnt = (char*) malloc(num_bytes);

	err = cudaMalloc((void**) &d_cnt, num_bytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device mem (error code %s)!\n", 
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	if (d_cnt == 0 || h_cnt == 0) {
		printf("couldn't allocate memory\n");
		return 1;
	}

	//int granularity = 1000;  // 0 means 'coarse'. full block granularity
	// Note COARSE_BLOCK=0 does not work well, due to multiples

	err = cudaMemset(d_cnt, 0, num_bytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy 0 to device mem (error code %s)!\n", 
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("totThreads=%d blocksPerGrid=%d threadsPerBlock=%d\n", totThreads, blocksPerGrid, threadsPerBlock);
	dim3 grid, block;
	grid.x = blocksPerGrid;
	//Note: max blocksPerGrid is 65535, then launch failure 'invalid argument'
	block.x = threadsPerBlock;
	//Note: max threadsPerBlock is 1024, then launch failure 'invalid configuration argument'

	start = clock();
	
    if (MAPPING == 1)
		kernel_1to1<<<grid, block>>>(d_cnt, topN);
    if (MAPPING == 2)
		kernel_2to1<<<grid, block>>>(d_cnt, topN);
	if (MAPPING == 3)
		kernel_6to2<<<grid, block>>>(d_cnt, topN);

	err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", 
        	cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(h_cnt, d_cnt, num_bytes, cudaMemcpyDeviceToHost);  // dest_ptr, src_ptr
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy device mem to host mem (error code %s)!\n", 
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 1to1 map: n = idx + 2  {idx:0..k => 2,3..k+2(=topN)}
	// 2to1 map: n = 2*idx + 3  {idx:0..k => 3,5,...2+1..2k+3(=topN)
	// 6to2 map: n = 6*idx/2 + 2*(idx%2) + 5  {idx:0,1,2,3,..k => 5,7,11,13,3k+2*(k%2)+5   // x3, 5, 7, x9, 11, 13, x15
	long total_prime_count = 0;
    if (MAPPING == 2 && topN >= 2)
		total_prime_count += 1;
	if (MAPPING == 3 && topN == 2)
		total_prime_count += 1;
	if (MAPPING == 3 && topN >= 3)
		total_prime_count += 2;

	for (long i = 0; i < totThreads; i++) {
	    total_prime_count += h_cnt[i];
	}

	// get end timestamp
    //gettimeofday(&end_time, NULL);
    //long delta_secs = end_time.tv_sec - start_time.tv_sec;
    //long delta_usecs = end_time.tv_usec - start_time.tv_usec;
    //double dbl_secs = delta_secs + (double)delta_usecs / 1000000.0;
    //printf("Total:   Primes=%d ExecTime=%f secs\n", total_prime_count, dbl_secs);
	end = clock();
	//cpuTime = (end-start) * 1.0 / CLOCKS_PER_SEC;
	cpuTime = difftime(end, start) / CLOCKS_PER_SEC;
    printf("Total:   Primes=%d ExecTime=%.4f secs\n", total_prime_count, cpuTime);

    if (DEBUG) {
		printf("h_cnt[] = ");
		for (long i = 0; i < totThreads; i++) {
			printf("%d ", h_cnt[i]);
		}
		printf("\n");
	}
    //printf("Total:   Primes=%d\n", total_prime_count);

	free(h_cnt);
	cudaFree(d_cnt);

	return 0;
}
