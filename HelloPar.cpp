#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#define NUM_THREADS 8

void* hello(void* arg) {
	printf("Hello from thread %i\n", (int)arg);
}

int main() {
	pthread_t thread[NUM_THREADS];
	int status;
	int i;
	for (i = 0; i < NUM_THREADS; i++) {
		if ( pthread_create(&thread[i], NULL, hello, (void*)i) != 0 ) {
			printf("pthread_create() error");
			exit(EXIT_FAILURE);
		}
	}
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], NULL);
	}
}
