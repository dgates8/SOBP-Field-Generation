#ifndef COIN_H
#define COIN_H
#include <string>
#include "sha256.h"

class Trick{
	private:
		bool spent;
		std::string coin;
	public:
		Trick();
		Trick(bool spent, std::string coin);
		bool get_spent();
		std::string get_coin();
};

#endif
