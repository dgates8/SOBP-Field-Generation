CC=nvcc
CFLAGS=-g -arch=sm_37 -I. 
DEPS = block.h sha256.h

%.o: %.c $(DEPS) 
	$(CC) -o $@ $< $(CFLAGS)

blockchain: blockchain.cu block.cpp sha256.cpp
	$(CC) -o blockchain blockchain.cu block.cpp sha256.cpp $(CFLAGS)

