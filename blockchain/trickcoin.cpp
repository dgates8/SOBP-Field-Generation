#include <iostream>
#include <iomanip>
#include <math.h>
#include <ctime>
#include "block.h"
#include <sstream>
#include <cstdlib>
#include "sha256.h"
#include "sha256.hh"
#include "coin.h"
#include "wallet.h"
#include <stdio.h>
#include <vector>
#include <string>

int main(int argc, char** argv){
	if(argc = 0){
		std::cout << "Please select an option or run -h for help" << std::endl;
		exit(1);
	}
	std::string str(argv[1]);
 
	if(str == "-h"){ 
			std::cout << "--gpu runs gpu accelerated miner" << std::endl;
			std::cout << std::endl;
			std::cout << "--cpu runs cpu miner" << std::endl;
			std::cout << std::endl;
			std::cout << "-b returns available balance" << std::endl;
			std::cout << std::endl;
			std::cout << "-t opens transfer options" << std::endl;
			std::cout << std::endl;
	}else if(str == "--gpu"){
			system("./mine GO");
	}else if(str == "--cpu"){
			system("./mine GO");
	}else if(str == "-b"){
			Wallet wallet;
			int balance = wallet.check_balance();		
			std::cout << "Balance is: " << balance << std::endl;	
	}else if(str == "-t"){
			Wallet wallet;
			wallet.transfer();
	}else{
		std::cout << "Invalid Selection, type -h for help" << std::endl;
		exit(1);
	}
}
