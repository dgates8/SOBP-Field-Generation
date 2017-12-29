#include "block.h"
#include <sstream>
#include <fstream>

Block::Block() {}

Block::Block(int i, std::string t, std::string h, std::string d): index(i), time(t), hash(h), data(d) {}

Block::Block(int i, std::string t, std::string h, std::string d, std::vector<Trick> c): index(i), time(t), hash(h), data(d) , coins(c) {}

std::string Block::hash_block(int index, std::string time, std::string hash, std::string data){
	std::stringstream ss;
	ss << index << time << hash << data;
	std::string key = ss.str();
	return sha256(key);

} 

void Block::set_coins(std::vector<Trick> coin_list){
	coins = coin_list;
}

std::string Block::generate_data(){
	std::stringstream ss;
	for(int i = 0; i < 5; i++){
		ss << coins[i].get_spent() << coins[i].get_coin();
	}
	std::string key = ss.str();
	return sha256(key);
}

void Block::write(){
	std::ofstream ofile("Trick.dat" , std::ios::app);
	ofile << index    << std::endl
	      << time     << std::endl
	      << hash     << std::endl
	      << data     << std::endl;
	      
	for(int i = 0; i < 5; i++){
	ofile << coins[i].get_spent() << "    "
	      << coins[i].get_coin()  << std::endl;
	}
	ofile << " ";
}

void Block::out(){
	std::cout << "Index:     " << index    << std::endl 
		  << "Time:      " << time     << std::endl
		  << "Hash:      " << hash     << std::endl
	          << "Data:      " << data     << std::endl;
	for(int i = 0; i < 5; i++){
	std::cout << "Coins:     " << coins[i].get_spent() << "  "
				   << coins[i].get_coin() << std::endl;
		  }
}

int Block::get_index()             {return index;}
std::string Block::get_time()      {return time;}
std::string Block::get_hash()      {return hash;}
std::string Block::get_data()      {return data;}

 
std::vector<Block> generateBlockchain(){
	std::ifstream ifile("Trick.dat", std::ios::in);
	std::vector<Block> blockchain;
	if(!ifile){
		std::cout << "Download Blockchain at: www.github.com/dgates8/ep/blockchain/Trick.dat" << std::endl;
	}else{		
		int index;
		bool spent;
		std::string time, date, hms, hash, data, coin_id, empty; 
		std::vector<Trick> coin_list(5);
		ifile >> index;
		ifile >> hms >> date;
		ifile >> hash;
		ifile >> data;
	        for(int i = 0; i < 5; i++){
			ifile >> spent >> coin_id;
			Trick coin(spent, coin_id);
			coin_list[i] = coin;
		}
		while(!ifile.eof()){
			getline(ifile,empty);
			time = hms + " " + date;
			Block block(index, time, hash, data, coin_list);
			blockchain.push_back(block);
			ifile >> index;
			ifile >> hms >> date;
			ifile >> hash;
			ifile >> data;
			for(int i = 0; i < 5; i++){
		                ifile >> spent >> coin_id;
				Trick coin(spent, coin_id);
				coin_list[i] = coin;
			}	
		};
		ifile.close();
		return blockchain;
	}
}

bool validateBlock(Block current, Block previous){
	std::string isValid = previous.hash_block(previous.get_index(), 
						  previous.get_time(),
						  previous.get_hash(),
						  previous.get_data());
	
	return (current.get_hash() == isValid);
}

