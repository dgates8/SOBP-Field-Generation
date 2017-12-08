#include <iostream>
#include <iomanip>
#include <math.h>
#include <ctime>
#include "block.h"
#include <sstream>
#include <cstdlib>
#include "sha256.h"
#include "sha256.hh"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define difficulty 3
#define blocks 2

//macro for error checking
#define cudaCheckError(){	   												  \
	cudaError_t err = cudaGetLastError();											  \
	if(err != cudaSuccess){              											  \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										  \
	}                                     											  \
}

//Cuda version  
__device__ const unsigned int sha256_key[64] = //UL = uint32
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
             0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
             0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
             0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
             0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
             0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
             0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
             0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
             0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
 
__host__ __device__ void SHA256Device::transform(char *message, unsigned int block_nb)
{
	uint32 w[64];
	uint32 wv[8];
	uint32 t1, t2;
	char *sub_block;
	int i;
	int j;
	for(i = 0; i < (int) block_nb; i++){
	sub_block = message + (i << 6);
		for(j = 0; j < 16; j++){
		    SHA2_PACK32(&sub_block[j << 2], &w[j]);
		}
		for(j = 16; j < 64; j++){
		    w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
		}
		for(j = 0; j < 8; j++){
		    wv[j] = m_h[j];
		}
		for(j = 0; j < 64; j++){
		    t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
			+ sha256_key[j] + w[j];
		    t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
		    wv[7] = wv[6];
		    wv[6] = wv[5];
		    wv[5] = wv[4];
		    wv[4] = wv[3] + t1;
		    wv[3] = wv[2];
		    wv[2] = wv[1];
		    wv[1] = wv[0];
		    wv[0] = t1 + t2;
		}
		for(j = 0; j < 8; j++){
		    m_h[j] += wv[j];
		}
	}
}
 
__host__ __device__ void SHA256Device::init()
{
	m_h[0] = 0x6a09e667;
	m_h[1] = 0xbb67ae85;
	m_h[2] = 0x3c6ef372;
	m_h[3] = 0xa54ff53a;
	m_h[4] = 0x510e527f;
	m_h[5] = 0x9b05688c;
	m_h[6] = 0x1f83d9ab;
	m_h[7] = 0x5be0cd19;
	m_len = 0;
	m_tot_len = 0;
}
 
__host__ __device__ void SHA256Device::update(char *message, unsigned int len)
{
	unsigned int block_nb;
	unsigned int new_len, rem_len, tmp_len;
	char *shifted_message;
	tmp_len = SHA224_256_BLOCK_SIZE - m_len;
	rem_len = len < tmp_len ? len : tmp_len;
	memcpy(&m_block[m_len], message, rem_len);
	if(m_len + len < SHA224_256_BLOCK_SIZE){
		m_len += len;
		return;
	}
	new_len = len - rem_len;
	block_nb = new_len / SHA224_256_BLOCK_SIZE;
	shifted_message = message + rem_len;
	transform(m_block, 1);
	transform(shifted_message, block_nb);
	rem_len = new_len % SHA224_256_BLOCK_SIZE;
	memcpy(m_block, &shifted_message[block_nb << 6], rem_len);
	m_len = rem_len;
	m_tot_len += (block_nb + 1) << 6;
}
 
__host__ __device__ void SHA256Device::final(char *digest)
{
	unsigned int block_nb;
	unsigned int pm_len;
	unsigned int len_b;
	int i;
	block_nb = (1 + ((SHA224_256_BLOCK_SIZE - 9)
		     < (m_len % SHA224_256_BLOCK_SIZE)));
	len_b = (m_tot_len + m_len) << 3;
	pm_len = block_nb << 6;
	memset(m_block + m_len, 0, pm_len - m_len);
	m_block[m_len] = 0x80;
	SHA2_UNPACK32(len_b, m_block + pm_len - 4);
	transform(m_block, block_nb);
	for(i = 0 ; i < 8; i++){
		SHA2_UNPACK32(m_h[i], &digest[i << 2]);
	}
}

__host__ __device__ void concatNonceToId(char* input, char* nonce){
	int index1 = 0;
	int index2 = 0;
	char concat[300] = {'0'};
	while(input[index1] != '\0'){
		concat[index1] = input[index1];
		index1++;
	}
	while(nonce != '\0'){
		concat[index1] = nonce[index2];
		index1++;
		index2++;
	}

	memcpy(input, concat, (index1+index2)*sizeof(char));
} 
 

__device__ char* convertNumberIntoArray(int number) {
    unsigned int length = (int)(log10((float)number)) + 1;
    char* arr = (char *) malloc(length * sizeof(char)), * curr = arr;
    do {
        *curr++ = number % 10;
        number /= 10;
    } while (number != 0);
    return arr;
}

__device__ void sha256_device(char* input, char* out, int idx){
	//get array size for input "string"
	int length = sizeof(input)/sizeof(0[input]);
	
	char digest[SHA256::DIGEST_SIZE];
	memset(digest,0,SHA256::DIGEST_SIZE);

	SHA256Device ctx = SHA256Device();
	ctx.init();
	ctx.update(input,length);
	ctx.final(digest);

	char buf[2*SHA256::DIGEST_SIZE+1];
	buf[2*SHA256::DIGEST_SIZE] = 0;
	for(int i = 0; i < SHA256::DIGEST_SIZE; i++){
		*(buf+i*2) = digest[i];
	}
	memcpy(out+idx*64, buf, 2*SHA256::DIGEST_SIZE);
	
}

//Solve the first five values of the hash for each id
__global__ void mine(char* id, char* hash_list, char *nonce, int* flag, int loop){
	int idx         = blockIdx.x*blockDim.x + threadIdx.x;
	int nonce_value = idx + loop*blockDim.x*threadIdx.x;
	nonce = convertNumberIntoArray(nonce_value);
	concatNonceToId(id, nonce);
	sha256_device(id, hash_list, idx);		
	
	int count = 0;
	#pragma unroll
	for(int i = 0; i < difficulty; i++){
		if(hash_list[i + idx*64] == '0'){
			count++;
		}	
	}
	if(count >= difficulty){
		atomicAdd(&flag[0],1);
	}
}


//generate random id string
void generate_random_id(char *s, const int len) {
	srand(time(NULL));
	char alphanum[] ="0123456789abcdefghijklmnopqrstuvwxyz";
	for (int i = 0; i < len; ++i) {
	s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}

	s[len] = 0;
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
	//set device
	cudaSetDevice(15);

	char *genesis_data;
	genesis_data = (char*)malloc(64*sizeof(char));
	generate_random_id(genesis_data,64);

	Block genesis(0, "0", "14634j64l5k7j547kj546k89679789k3j5j1k9d9f9gt928e7f3838d3d820dg1f", "1hfuefh3uhr3urh3jrn3kd3rpdafksdlfkao79999999fe999999333hyui3id3d", genesis_data);
	Block blockchain[10];
	blockchain[0] = genesis;
	blockchain[0].out();
	std::cout << std::endl;
	
	//flag arrays for loop termination
	int *h_flag, *d_flag; 
	h_flag = (int*)malloc(5*sizeof(int));

	//cuda allocation stuff
	char *d_id, *d_hash, *d_hash_list, *d_nonce;

	cudaMalloc((void**)&d_id, 264*sizeof(char));
	cudaCheckError();
	cudaMalloc((void**)&d_flag, 5*sizeof(int));
	cudaCheckError();
	cudaMalloc((void**)&d_hash, 64*sizeof(char));
	cudaCheckError();
	cudaMalloc((void**)&d_hash_list, 2*64000*64*sizeof(char));
	cudaCheckError();
	cudaMalloc((void**)&d_nonce, 200*sizeof(char));
	cudaCheckError();
		
	cudaMemcpy(d_id, genesis_data, 64*sizeof(char), cudaMemcpyHostToDevice);
	cudaCheckError();
	
	//loop for nonce values
	int loop = 0;

	for(int i = 1; i < blocks; i++){
		do{	
			//mine next block 
			mine<<<1000,64>>>(d_id, d_hash_list, d_nonce, d_flag, loop);
			
			//load flag to see if new block was found
			cudaMemcpy(h_flag, d_flag, 5*sizeof(int), cudaMemcpyDeviceToHost);
			cudaCheckError();
		
			//update loop for next set of values
			loop++;
	
		}while(h_flag[0] == 0);	
		
		std::cout << "Blocked Mined successfully with nonce: " << h_flag[1] << std::endl;

		//validate next block and then update id for next mining block where id is transaction amount
		Block nextBlock = next(blockchain[i-1]);
		blockchain[i] = nextBlock;
		blockchain[i].out();
		std::cout << std::endl;
		
		char* nextId = const_cast<char*>(blockchain[i].get_data().c_str());

		//update next block id for proof of work step
		cudaMemcpy(d_id, nextId, 64*sizeof(char), cudaMemcpyHostToDevice);
		cudaCheckError();
	}
}
