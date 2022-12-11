#pragma once
#include <iostream>
#include <vector>
#include "data_loader.h"

void combinationUntil(std::vector<int> &indices, int n, int r, int index, int data[], int i, std::vector<std::vector<int>> &combinations);

void find_combinations(std::vector<int> &indices, int n, int r, std::vector<std::vector<int>> &combinations);