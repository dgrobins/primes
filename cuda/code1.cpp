#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>

int main()
{
	int dimx = 16;
	int num_bytes= dimx*sizeof(int);

	int *d_a=0, *h_a=0; // device and host pointers

	h_a = (int*) malloc(num_bytes);
	cudaMalloc((void**) &d_a, num_bytes);

	if (h_a == 0 || d_a == 0)
	{
		printf("couldn't allocate memory\n");
		return 1;
	}

	cudaMemset(d_a, 0, num_bytes);
	cudaMemcpy(h_a, d_a, num_bytes, cudaMemcpyDeviceToHost);  // dest_ptr, src_ptr, direction (dev2host)

	for (int i=0; i < dimx; i++)
		printf("%d ", h_a[i]);
	printf("\n");

	free(h_a);
	cudaFree(d_a);

	return 0;
}
