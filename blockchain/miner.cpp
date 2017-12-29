#include <iostream>
#include <iomanip>
#include <math.h>
#include <ctime>
#include "block.h"
#include <sstream>
#include <cstdlib>
#include "sha256.h"
#include "coin.h"
#include <stdio.h>
#include <vector>

#define difficulty 6
#define blocks 1

//macro for error checking
#define cudaCheckError(){	   												  \
	cudaError_t err = cudaGetLastError();											  \
	if(err != cudaSuccess){              											  \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										  \
	}                                     											  \
}

//Solve the first five values of the hash for each id
void mine(std::string hash, int* flag, int loop, std::string &final_hash){
	int count = 0;
	long long nonce = loop;
	std::string s = std::to_string(nonce);
	std::string hash_to_check = hash + s;
	std::string nonced_hash = sha256(hash_to_check);
	for(int i = 0; i < difficulty; i++){
		if(nonced_hash[i] == '0'){
			count++;
		}
	}
	if(count >= difficulty){
		flag[0] = 1;
		flag[1] = nonce;
		final_hash = nonced_hash;
	}
}




//generate random id string
void generate_random_id(char *s, const int len) {

	char alphanum[] ="0123456789abcdefghijklmnopqrstuvwxyz";
	for (int i = 0; i < len; ++i) {
	s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}

	s[len] = 0;
}

std::vector<Trick> generate_coins(Block block){
	
	bool spent = false;
	std::string coin_list[5];
	std::vector<Trick> coins;
	for(int i = 0; i < 5; i++){
		char *key;	
		key = (char*)malloc(64*sizeof(char));
		generate_random_id(key, 64);
		coin_list[i] = sha256(key);
		Trick coin(spent, coin_list[i]);
		coins.push_back(coin);
	}
	return coins;
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

	//generate coins before hashing block
	std::vector<Trick> coins = generate_coins(current);
	
	//gnerate data and hash
	current.set_coins(coins);
	std::string data = current.generate_data();
	std::string hash = current.hash_block(current.get_index(), 
					      current.get_time(), 
					      current.get_hash(),  
					      current.get_data());
	
	//set new block
	Block nextBlock(index, str, hash, data, coins);
	return nextBlock;
}

int main(int argc, char** argv){
	srand(time(NULL));
	std::string str(argv[1]);
	std::vector<Block> blockchain;
	blockchain = generateBlockchain();

	if(blockchain[0].get_index() != 0){	
		system("rm *.dat");
		blockchain.clear();	
		
		//time
		time_t rawtime;
		struct tm *timeinfo;
		char buffer[80];

		time (&rawtime);
		timeinfo = localtime(&rawtime);

		strftime(buffer,sizeof(buffer),"%d-%m-%Y %I:%M:%S",timeinfo);
		std::string str(buffer);

		std::string genesis_key = sha256(argv[1]);
		Block genesis(0, buffer, sha256(genesis_key), genesis_key.c_str());
		std::vector<Trick> coins = generate_coins(genesis);
		genesis.set_coins(coins);

		genesis.out();
		genesis.write();
		std::cout << std::endl;
		blockchain = generateBlockchain();
	}
	
	int idx = blockchain.back().get_index();
	blockchain[idx].out();
	if(!(validateBlock(blockchain[idx], blockchain[idx-1]))){;
		std::cout << "Blockchain has been modified" << std::endl;
		exit(1);
	}
	idx++;

	//loop for nonce values
	for(int i = idx; i < idx+blocks; i++){
		int loop = 0;	
		std::string hash;
		int h_flag[2] = {0,0};
		do{	
			//mine next block 
			mine(blockchain[i-1].get_hash(), h_flag, loop, hash);
			loop++;

		}while(h_flag[0] == 0);	
			std::cout << std::endl;
			//copy off hash that should solve the problem, need to validate it before mining next block	
			std::cout << "Block Mined with nonce: " << h_flag[1] << std::endl;				
			std::cout << "Hash for nonce: " << hash << std::endl; 
			std::cout << std::endl;
			
	
		//validate next block and then update id for next mining block where id is transaction amount
		Block nextBlock = next(blockchain[i-1]);
		blockchain.push_back(nextBlock);
		blockchain[i].out();
		std::cout << std::endl;
		
		if(validateBlock(blockchain[i], blockchain[i-1]) && i != 1){
			blockchain[i].write();
		}
	}
}
