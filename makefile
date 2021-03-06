CC=nvcc	
CFLAGS=-g -std=c++11 -03 -arch=sm_37

all: generate write sum
generate: generateBeams.cu
	$(CC) -o generate generateBeams.cu $(CFLAGS)
write: writeDepthDose.cu
	$(CC) -o write writeDepthDose.cu $(CFLAGS)
sum: sumFieldsForSOBP.cu
	$(CC) -o sum sumFieldsForSOBP.cu $(CFLAGS)

