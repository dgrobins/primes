c1: Prime1.exe
	Prime1 1
	Prime1 2
	Prime1 3
	Prime1 4
	Prime1 5
	Prime1 6
	Prime1 7
	#ifdef LONG
	#Prime1 8

c2: PrimeList.exe
	PrimeList 1
	PrimeList 2
	PrimeList 3
	PrimeList 4
	PrimeList 5
	PrimeList 6
	PrimeList 7
	PrimeList 8

cp1: PrimeParBlock.exe
	PrimeParBlock 1
	PrimeParBlock 2
	PrimeParBlock 3
	PrimeParBlock 4
	PrimeParBlock 5
	PrimeParBlock 6
	PrimeParBlock 7
	PrimeParBlock 8

cp7: PrimeParBlock.exe
	PrimeParBlock 7

j1: Prime1.class
	java Prime1 1
	java Prime1 2
	java Prime1 3
	java Prime1 4
	java Prime1 5
	java Prime1 6
	java Prime1 7

j2: PrimeList.class
	java PrimeList 1
	java PrimeList 2
	java PrimeList 3
	java PrimeList 4
	java PrimeList 5
	java PrimeList 6
	java PrimeList 7
	java PrimeList 8

jx1: Prime1_x32.class
	java Prime1_x32 1
	java Prime1_x32 2
	java Prime1_x32 3
	java Prime1_x32 4
	java Prime1_x32 5
	java Prime1_x32 6
	java Prime1_x32 7
	#java Prime1_x32 8

jp1: PrimeParBlock.class
	java PrimeParBlock 1
	java PrimeParBlock 2
	java PrimeParBlock 3
	java PrimeParBlock 4
	java PrimeParBlock 5
	java PrimeParBlock 6
	java PrimeParBlock 7
	#java PrimeParBlock 8

jd1: PrimeParDynamic.class
	java PrimeParDynamic 1
	java PrimeParDynamic 2
	java PrimeParDynamic 3
	java PrimeParDynamic 4
	java PrimeParDynamic 5
	java PrimeParDynamic 6
	java PrimeParDynamic 7
	#java PrimeParDynamic 8

p1: 
	Prime1.py 1
	Prime1.py 2
	Prime1.py 3
	Prime1.py 4
	Prime1.py 5
	Prime1.py 6
	#ifdef LONG
	#Prime1.py 7

p2: 
	PrimeList.py 1
	PrimeList.py 2
	PrimeList.py 3
	PrimeList.py 4
	PrimeList.py 5
	PrimeList.py 6
	PrimeList.py 7

h1: hello.exe
	hello

hp1: HelloPar.exe
	HelloPar

hello.exe: hello.cpp
	gcc -o hello hello.cpp

HelloPar.exe: HelloPar.cpp
	gcc -o HelloPar HelloPar.cpp

Prime1.exe: Prime1.cpp
	g++ -o Prime1 -O3 Prime1.cpp

PrimeList.exe: PrimeList.cpp
	g++ -o PrimeList -O3 PrimeList.cpp

PrimeParBlock.exe: PrimeParBlock.cpp
	g++ -o PrimeParBlock -O3 PrimeParBlock.cpp

Prime1.class: Prime1.java
	javac Prime1.java

PrimeList.class: PrimeList.java
	javac PrimeList.java

Prime1_x32.class: Prime1_x32.java
	javac Prime1_x32.java

PrimeParBlock.class: PrimeParBlock.java
	javac PrimeParBlock.java

PrimeParDynamic.class: PrimeParDynamic.java
	javac PrimeParDynamic.java

clean:
	rm *.class *.exe
