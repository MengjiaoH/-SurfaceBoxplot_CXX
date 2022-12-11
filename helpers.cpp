#include "heplers.h"


void combinationUntil(std::vector<int> &indices, int n, int r, int index, int data[], int i, std::vector<std::vector<int>> &combinations)
{
    if (index == r){
        std::vector<int> temp(r, 0);
        for (int j = 0; j < r; j++){
            // std::cout << data[j] << " ";
            temp[j] = data[j];
        }
        combinations.push_back(temp);
        return;
    }
    if(i >= n){
        return;
    }
    data[index] = indices[i];
    combinationUntil(indices, n, r, index+1, data, i+1, combinations);
    combinationUntil(indices, n, r, index, data, i+1, combinations);


}
void find_combinations(std::vector<int> &indices, int n, int r, std::vector<std::vector<int>> &combinations)
{
    int data[r];

    combinationUntil(indices, n, r, 0, data, 0, combinations);

}