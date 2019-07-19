#include "half.hpp"
#include <iostream>
#include <cmath>

int main()
{
	half_precision::half a;
	half_precision::half b=2.2;

	std::cout << std::exp(b) << std::endl;
}