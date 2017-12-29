#include "wallet.h"
#include <fstream>
#include <cstdlib>

Wallet::Wallet() {}
Wallet::Wallet(int b) : balance(b) {}

void Wallet::deposit_coins(std::vector<Trick> coin_list){
	std::ofstream ofile("wallet.dat", std::ios::app);
	for(int i = 0; i < 5; i++){
		ofile << coin_list[i].get_coin() << std::endl;
	}	
}

int Wallet::check_balance(){
	std::ifstream ifile("wallet.dat");
	std::vector <std::string> coins;
	std::string coin;
	if(!ifile){
		std::cout << "No wallet found" << std::endl;
		exit(0);
	}
	while(!ifile.eof()){
		ifile >> coin;
		coins.push_back(coin);
	}
	ifile.close();
	return coins.size();
}

void Wallet::validateCoins(std::vector<std::string> coins){
	std::ifstream ifile("transfer.dat");
	std::string coin;
	if(!ifile){
		std::cout << "No file found" << std::endl;
		exit(0);
	}	
	while(!ifile.eof()){
		ifile >> coin;
		coins.push_back(coin);
	}
	for(int i = 0; i < coins.size(); i++){
		coins[i] = sha256(coins[i]);
	}
	ifile.close();

	std::ifstream infile("Trick.dat");
	std::vector<Trick> coin_list;

	if(!infile){
		std::cout << "File not found, try downloading blockchain again" << std::endl;
		exit(0);
	}else{		
		int index;
		bool spent;
		std::string time, date, hms, hash, data, coin_id, empty; 
		infile >> index;
		infile >> hms >> date;
		infile >> hash;
		infile >> data;
	        for(int i = 0; i < 5; i++){
			infile >> spent >> coin_id;
			Trick coin(spent, coin_id);
			coin_list.push_back(coin);
		}
		while(!ifile.eof()){
			getline(ifile,empty);
			time = hms + " " + date;
			infile >> index;
			infile >> hms >> date;
			infile >> hash;
			infile >> data;
			for(int i = 0; i < 5; i++){
		                infile >> spent >> coin_id;
				Trick coin(spent, coin_id);
				coin_list.push_back(coin);
			}	
		}
	}
	std::vector<std::string> trickcoin;
	for(int i = 0; i < coin_list.size(); i++){
		trickcoin.push_back(coin_list[i].get_coin());
	}
	
	std::vector<int> flag;
	for(int i = 0; i < coins.size(); i++){
		for(int j = 0; j < trickcoin.size(); j++){
			if(coins[i] == trickcoin[j]){
				flag.push_back(1);	
			}
		}
	}

	if(flag.size() != coins.size()){
		std::cout << (coins.size() - flag.size()) << " coins are invalid" << std::endl;
		std::cout << "Transfer failed" << std::endl;
		system("rm transfer.dat");
	}
	
}

void Wallet::transfer(){
	
	//read in coins to object and then transfer coins to user and delete from wallet sent coins.
	std::ifstream ifile("wallet.dat");
	std::vector <std::string> coins;
	std::string coin;
	if(!ifile){
		std::cout << "No wallet found" << std::endl;
		exit(1);
	}
	while(!ifile.eof()){
		ifile >> coin;
		coins.push_back(coin);
	}
	ifile.close();
	
	std::string public_ip, transfer_ip;
	int number_to_transfer;
	std::cout << "Enter IP for location of transfer" << std::endl;
	std::cin >> transfer_ip;
	std::cout << "Enter number of coins to transfer" << std::endl;
	std::cin >> number_to_transfer;

	std::ofstream ofile("transfer.dat");
	std::ofstream outfile("wallet.dat");
	for(int i = 0; i < number_to_transfer; i++){
		try{
			ofile << coins[i] << std::endl;
		}catch(...){
			std::cout << "Not enough coins" << std::endl;
			exit(0);
		}
	}
	ofile.close();

	validateCoins(coins);

	for(int j = number_to_transfer; j < coins.size(); j++){
		outfile << coins[j] << std::endl;
	}
	outfile.close();

	public_ip = system("ifconfig eth1 | awk '/inet / {print $2}' | sed 's#/.*##'");
	system(("nc -w 3 "+ transfer_ip +" < transfer.dat").c_str()); 
}

