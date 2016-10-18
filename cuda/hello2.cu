#include <stdio.h>

__global__ void mykernel(void) {
}

int main(void) {
	mykernel<<<1,1>>>();  // kernel launch
	printf("hellow World, Cuda with DeviceCode!\n");
	return 0;
}
