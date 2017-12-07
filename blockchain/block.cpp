#include "block.h"
#include <sstream>

Block::Block() {}

Block::Block(int i, std::string t, std::string h, std::string p, std::string d){
	index = i;
	time = t;
	hash = h;
	previous_hash = p;
	data = d;
}

std::string Block::hash_block(int index, std::string time, std::string hash, std::string previous_hash, std::string data){
	std::stringstream ss;
	ss << index << time << hash << previous_hash << data;
	std::string key = ss.str();
	return sha256(key);

} 

void Block::out(){
	std::cout << "Index:     " << index         << std::endl 
		  << "Time:      " << time          << std::endl
		  << "Hash:      " << hash          << std::endl
		  << "Data:      " << data          << std::endl
		  << "Previous:  " << previous_hash << std::endl;
}

int Block::get_index()                 {return index;}
std::string Block::get_time()          {return time;}
std::string Block::get_hash()          {return hash;}
std::string Block::get_previous_hash() {return previous_hash;}
std::string Block::get_data()          {return data;}
 

