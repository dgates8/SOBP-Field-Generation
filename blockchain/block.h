#include <iostream>
#include "sha256.h"
#include <time.h>
#include <string>

class Block{
	private:
		int index;
		std::string time, previous_hash, hash, data;
	public:
		Block();
		Block(int index, std::string time, std::string hash, std::string previous_hash, std::string data);
		std::string hash_block(int index, std::string time, std::string hash, std::string previous_hash, std::string data);	
		void out();
		int get_index();
		std::string get_time();
		std::string get_hash();
		std::string get_previous_hash();
		std::string get_data();
};
