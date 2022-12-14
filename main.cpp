#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <mutex>
#include <atomic> 
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include "data_loader.h"
#include "heplers.h"

using namespace std::chrono;
std::mutex depth_mutex;
std::mutex total_depth_mutex;

int main(int argc, const char** argv)
{
    std::string dataPath = argv[1]; // the path to the ensemble data 
    int n_members = std::atoi(argv[2]); // the number of memebers 
    const vec2i dims = vec2i(std::atoi(argv[3]), std::atoi(argv[4]));
    const std::string voxel_type = "float32";
    const int selected_members = std::atoi(argv[5]);
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

    
    std::vector<std::vector<int>> combinations_temp;
    // remove the index 
    std::vector<int> indices_temp(n_members);
    std::iota(indices_temp.begin(), indices_temp.end(), 0); 
    indices_temp.erase(indices_temp.begin() + 0);
    find_combinations(indices_temp, indices_temp.size(), selected_members, combinations_temp);
    int n_combinations = combinations_temp.size();
    std::cout << "Select " << selected_members << " members." << "\n";
    std::cout << "There are " << n_combinations << " combinations." << "\n";
    int n_voxels = volumes[0].n_voxels(); // number of voxels
    std::vector<int> totalDepthList(n_members, 0);
    
    auto start_calculation = high_resolution_clock::now();
    // Calculate the depth 
    /*
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
    */
   
   tbb::parallel_for(tbb::blocked_range3d<int> (0, n_members, 0, n_combinations, 0, n_voxels), [&](const tbb::blocked_range3d<int> &r){
        for(int n = r.pages().begin(); n < r.pages().end(); n++){
            // int totalDepth = totalDepthList[n];
            // std::atomic<int> totalDepth(0);
            Volume2D cur_volume = volumes[n];
            auto cur_voxels = *(cur_volume.voxel_data);
            // find combinations
            std::vector<std::vector<int>> combinations;
            // remove the index 
            std::vector<int> indices(n_members);
            std::iota(indices.begin(), indices.end(), 0); 
            indices.erase(indices.begin() + n);
            find_combinations(indices, indices.size(), selected_members, combinations);
            // std::vector<int> depths(combinations.size(), 0);
            
            for(int c = r.rows().begin(); c < r.rows().end(); c++){
                std::vector<int> combination = combinations[c];
                // std::cout << n << " " << combinations[c].x << " " << combinations[c].y << "\n";
                // int depth = depths[c];
                std::atomic<int> depth(0);
                // std::vector<int> voxel_depth(n_voxels, 0);
                for(int i = r.cols().begin(); i < r.cols().end(); i++){
                    float cur_data_value = cur_voxels[i];
                    float minval = 1000000;
                    float maxval = -1000000;
                    for(int j = 0; j < combination.size(); j++){
                        Volume2D volume = volumes[combination[j]];
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
                        // std::lock_guard<std::mutex> depth_guard(depth_mutex);
                        depth += 1;
                        // voxel_depth[i] += 1;
                    }
                } // end of all voxels
                // int d = std::accumulate(voxel_depth.begin(),voxel_depth.end(),0);
                // depths[c] += d;
                std::lock_guard<std::mutex> total_depth_guard(total_depth_mutex);
                totalDepthList[n] += depth;
            }// for all combinations
            // totalDepthList[n] += std::accumulate(depths.begin(),depths.end(),0);

        }
        

   }); // end of tbb
   
    auto stop_calculation = high_resolution_clock::now();
    auto duration__calculation = duration_cast<milliseconds>(stop_calculation - start_calculation);
    std::cout << "Calucating Depth costs " << duration__calculation.count() << " ms." << std::endl;
    
    for(int n = 0; n < n_members; n++){
        std::cout << n << ": " << totalDepthList[n] << "\n";
    }
    


    return 0;
}