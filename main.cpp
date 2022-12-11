#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "data_loader.h"
#include "heplers.h"

using namespace std::chrono;

int main(int argc, const char** argv)
{
    std::string dataPath = argv[1]; // the path to the ensemble data 
    int n_members = std::atoi(argv[2]); // the number of memebers 
    const vec2i dims = vec2i(std::atoi(argv[3]), std::atoi(argv[4]));
    const std::string voxel_type = "float32";
    const int selected_members = 2;
    // std::cout << dims.x << " " << dims.y << "\n";

    std::vector<Volume2D> volumes(n_members); // save all members in this vector 

    // Loading Ensemble Data 
    
    auto start_loading = high_resolution_clock::now();
    for(int n = 0; n < n_members; ++n){
        const std::string path = dataPath + "m_" + std::to_string(n) + ".raw";
        // std::cout << path << "\n";
        volumes[n] = load_raw_volume(path, dims, voxel_type);

    }
    auto stop_loading = high_resolution_clock::now();
    auto duration_loading = duration_cast<milliseconds>(stop_loading - start_loading);
    std::cout << "Loading ensember data costs " << duration_loading.count() << " ms." << std::endl;

    int n_voxels = volumes[0].n_voxels(); // number of voxels
    std::vector<int> totalDepthList(n_members, 0);
    
    auto start_calculation = high_resolution_clock::now();
    // Calculate the depth 
    for(int n = 0; n < n_members; n++){
        int totalDepth = 0;
        Volume2D cur_volume = volumes[n];
        auto cur_voxels = *(cur_volume.voxel_data);
        // find combinations
        std::vector<std::vector<int>> combinations;
        // remove the index 
        std::vector<int> indices(n_members);
        std::iota(indices.begin(), indices.end(), 0); 
        indices.erase(indices.begin() + n);
        find_combinations(indices, indices.size(), selected_members, combinations);
        // iterate the combinations
        for(int c = 0; c < combinations.size(); c++){
            std::vector<int> combination = combinations[c];
            // std::cout << n << " " << combinations[c].x << " " << combinations[c].y << "\n";
            int depth = 0;
            for(int i = 0; i < n_voxels; ++i){
                float cur_data_value = cur_voxels[i];
                float minval = 1000000;
                float maxval = -1000000;
                for(int r = 0; r < combination.size(); r++){
                    Volume2D volume = volumes[combination[r]];
                    auto voxels = *(volume.voxel_data);
                    float data_value = voxels[i];
                    if (data_value > maxval){
                        maxval = data_value;
                    }
                    if (data_value < minval){
                        minval = data_value;
                    }
                }
                // std::cout << i << " " << minval << " " << maxval << "\n";
                if ((cur_data_value >= minval) && (cur_data_value <= maxval)){
                    depth += 1;
                }
            }// end of for loop of voxels
            totalDepth = totalDepth + depth;
        } // end of all combinations
        totalDepthList[n] = totalDepth;
    } // end of all members

    auto stop_calculation = high_resolution_clock::now();
    auto duration__calculation = duration_cast<milliseconds>(stop_calculation - start_calculation);
    std::cout << "Calucating Depth costs " << duration__calculation.count() << " ms." << std::endl;

    for(int n = 0; n < n_members; n++){
        std::cout << n << ": " << totalDepthList[n] << "\n";
    }
    


    return 0;
}