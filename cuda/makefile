#
# GPU Nvidia GT 730M: ARCH = sm_30

p1: ParPrime.exe
	ParPrime 1
	ParPrime 2
	ParPrime 3
	ParPrime 4
	ParPrime 5
	ParPrime 6
	ParPrime 7
	#ParPrime 8

p8: ParPrime.exe
	ParPrime 8

ParPrime.exe: ParPrime.cu
	nvcc -arch sm_30 -o ParPrime ParPrime.cu

reg_usage: code2.cu
	nvcc -arch sm_30 --ptxas-options=-v code2.cu

pptx: ParPrime.cu
	nvcc --ptx -arch sm_30 ParPrime.cu

pdis: ParPrime.cu
	nvcc --cubin -arch sm_30 ParPrime.cu
	nvdisasm ParPrime.cubin > ParPrime.cuasm

c1: code1.exe
	code1

c2: code2.exe
	code2

h1: hello.exe
	hello

h2: hello2.exe
	hello2

code1.exe: code1.cpp
	nvcc -o code1 code1.cpp

code2.exe: code2.cu
	nvcc -o code2 code2.cu

c2ptx: code2.cu
	nvcc --ptx code2.cu

hello.exe: hello.cu
	nvcc -o hello hello.cu

hello2.exe: hello2.cu
	nvcc -o hello2 hello2.cu

