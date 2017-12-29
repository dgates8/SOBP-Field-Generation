#include "coin.h"
#include <sstream>

Trick::Trick() {}

Trick::Trick(bool s, std::string c) : spent(s), coin(c) {}

bool Trick::get_spent()             {return spent;}
std::string Trick::get_coin()       {return coin;}
