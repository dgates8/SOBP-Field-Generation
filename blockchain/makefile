NVCC_CHECK=$(shell which nvcc)
NVCC_TEST=$(notdir $(NVCC_CHECK))
ifeq ($(NVCC_TEST),nvcc)
CC=nvcc	
CFLAGS=-g -std=c++11 -I. 
DEPS = block.h sha256.h coin.h wallet.h
%.o: %.c $(DEPS) 
	$(CC) -o $@ $< $(CFLAGS)
all: mine trickcoin 
mine: miner.cu block.cpp sha256.cpp coin.cpp wallet.cpp 
	$(CC) -o mine miner.cu block.cpp sha256.cpp coin.cpp wallet.cpp $(CFLAGS)
trickcoin: trickcoin.cpp block.cpp sha256.cpp coin.cpp wallet.cpp
	$(CC) -o trickcoin trickcoin.cpp block.cpp sha256.cpp coin.cpp wallet.cpp $(CFLAGS)
else
CC=g++
CFLAGS=-g -std=c++11 -I. 
DEPS = block.h sha256.h coin.h wallet.h
%.o: %.c $(DEPS) 
	$(CC) -o $@ $< $(CFLAGS)
all: mine trickcoin
mine: miner.cpp block.cpp sha256.cpp coin.cpp wallet.cpp 
	$(CC) -o mine miner.cpp block.cpp sha256.cpp coin.cpp wallet.cpp $(CFLAGS)
trickcoin: trickcoin.cpp block.cpp sha256.cpp coin.cpp wallet.cpp
	$(CC) -o trickcoin trickcoin.cpp block.cpp sha256.cpp coin.cpp wallet.cpp $(CFLAGS)
endif

