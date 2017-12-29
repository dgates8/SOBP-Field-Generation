#ifndef WALLET_H
#define WALLET_H
#include "sha256.h"
#include "block.h"
#include "coin.h"
#include <string>
#include <vector>
#include <sstream>

class Wallet{
	private: 
		int balance;
		std::vector<Trick> coins;
	public:
		Wallet();
		Wallet(int balance);
		void deposit_coins(std::vector<Trick> coins);
		int  check_balance();
		void validateCoins(std::vector<std::string> coins);
		void transfer();
};

#endif
