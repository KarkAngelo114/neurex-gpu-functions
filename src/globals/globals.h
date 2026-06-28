#pragma once
#include <vector>
using Array = std::vector<float>;

bool get_Global_Boolean_On_GPU();
const Array& getGlobalWeights(int pointer);
const Array& getGlobalBiases(int pointer);
const Array& getGlobalOutputTensors(int pointer);