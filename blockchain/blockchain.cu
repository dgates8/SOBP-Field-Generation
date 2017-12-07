#include <iostream>
#include <iomanip>
#include <math.h>
#include <ctime>
#include "block.h"
#include <sstream>
#include <cstdlib>
#include "sha256.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define difficult 6
#define blocks 4

//macro for error checking
#define cudaCheckError(){	   												  \
	cudaError_t err = cudaGetLastError();											  \
	if(err != cudaSuccess){              											  \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										  \
	}                                     											  \
}

//key to checking on the GPU if two hashes are equal. add the values for each random has and then compare
__inline__ __device__ int ASCIatoiSum(char *s, const int difficulty){
	int key = 0;
	for(int i = 0; i < difficulty; i++){
		key += int(s[i]);
	}
	return key;
}

//device verison 
__device__ char* generate_random_hash(char* s, const int len, const int seed, const int idx) {
	const char alphanum[] ="0123456789abcdefghijklmnopqrstuvwxyz";
	curandState_t state;
	curand_init(seed, idx, 0, &state);

	for (int i = 0; i < len; ++i) {
		s[i] = alphanum[curand(&state) % (sizeof(alphanum) - 1)];
	}

	s[len] = 0;
	return s;
}


//generate random id string
void generate_random_id(char *s, const int len) {
	const char alphanum[] ="0123456789abcdefghijklmnopqrstuvwxyz";
	for (int i = 0; i < len; ++i) {
	s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}

	s[len] = 0;
}


//Solve the first five values of the hash for each id
__global__ void mine(char* id, int* flag, char* hash, int difficulty, int* seed){
	int idx = blockIdx.x*blockDim.x	+ threadIdx.x;
	int id_value   = ASCIatoiSum(id, difficulty);
	int hash_value = ASCIatoiSum(generate_random_hash(hash, 64, seed[idx], idx), difficulty);
	if(id_value == hash_value){
		flag[0] = 1;
	}	
}

//blockchain
Block next(Block current){
	
	//index
	int index = current.get_index() + 1;

	//time
	time_t rawtime;
	struct tm *timeinfo;
	char buffer[80];

	time (&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer,sizeof(buffer),"%d-%m-%Y %I:%M:%S",timeinfo);
	std::string str(buffer);

	//hash
	std::string hash = current.hash_block(current.get_index(), 
					      current.get_time(), 
					      current.get_hash(), 
					      current.get_previous_hash(), 
					      current.get_data());
	//data
	char *data;
	data = (char*)malloc(64*sizeof(char));
	generate_random_id(data, 64);

	//set new block
	Block nextBlock(index, str, hash, current.get_hash(), data);
	return nextBlock;
}

int main(){
	//set difficulty
	cudaSetDevice(15);

	char *genesis_data;
	genesis_data = (char*)malloc(64*sizeof(char));
	generate_random_id(genesis_data,64);
	genesis_data[0] = '0';
	genesis_data[1] = '0';
	genesis_data[2] = '0';

	Block genesis(0, "0", "14634j64l5k7j547kj546k89679789k3j5j1k9d9f9gt928e7f3838d3d820dg1f", "1hfuefh3uhr3urh3jrn3kd3rpdafksdlfkao79999999fe999999333hyui3id3d", genesis_data);
	Block blockchain[10];
	blockchain[0] = genesis;
	blockchain[0].out();
	std::cout << std::endl;

	//seeds for cuRAND
	int seed[640000];
	srand(time(NULL));
	for(int i = 0; i < 640000; i++){
		seed[i] = rand() % 100000;
	}
		
	//flag arrays for loop termination
	int *h_flag, *d_flag, *d_seed; 
	h_flag = (int*)malloc(5*sizeof(int));

	//cuda allocation stuff
	char *d_id, *d_hash;

	cudaMalloc((void**)&d_id, 64*sizeof(char));
	cudaCheckError();
	cudaMalloc((void**)&d_flag, 5*sizeof(int));
	cudaCheckError();
	cudaMalloc((void**)&d_hash, 64*sizeof(char));
	cudaCheckError();
	cudaMalloc((void**)&d_seed, 640000*sizeof(int));
	cudaCheckError();
		
	cudaMemcpy(d_id, genesis_data, 64*sizeof(char), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(d_seed, seed, 640000*sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();
	
	//number of loops to solve problem
	int loop = 0;

	for(int i = 1; i < blocks; i++){
		do{	
			mine<<<10000,64>>>(d_id, d_flag, d_hash, difficult, d_seed);
			cudaMemcpy(h_flag, d_flag, 5*sizeof(int), cudaMemcpyDeviceToHost);
			cudaCheckError();
			loop++;
			if(loop % 1000 == 0){
				std::cout << loop << std::endl;
			}
		}while(h_flag[0] == 0);	
		
		//validate next block and then update id for next mining block where id is transaction amount
		Block nextBlock = next(blockchain[i-1]);
		blockchain[i] = nextBlock;
		blockchain[i].out();
		std::cout << std::endl;
		
		const char *nextId = blockchain[i].get_data().c_str();
		
		//update next block id for proof of work step
		cudaMemcpy(d_id, nextId, 64*sizeof(char), cudaMemcpyHostToDevice);
		cudaCheckError();
	}
	
}

