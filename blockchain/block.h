#ifndef BLOCK_H
#define BLOCK_H

#include <iostream>
#include "sha256.h"
#include <time.h>
#include <string>
#include <vector>
#include "coin.h"

class Block{
	private:
		int index;
		std::string time, hash, data;
		std::vector<Trick> coins;
	public:
		Block();
		Block(int index, std::string time, std::string hash, std::string data);
		Block(int index, std::string time, std::string hash, std::string data, std::vector<Trick> coins);

		std::string hash_block(int index, std::string time, std::string hash, std::string data);	
		void set_coins(std::vector<Trick> coins);
		std::string generate_data();
		void out();
		void write();
		int get_index();
		std::string get_time();
		std::string get_hash();
		std::string get_data();
};
bool validateBlock(Block current, Block previous);
std::vector<Block> generateBlockchain();

#endif
