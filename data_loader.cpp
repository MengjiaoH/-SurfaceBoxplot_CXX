#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm>
#include "data_loader.h"



Volume2D load_raw_volume(const std::string &fname,
                         const vec2i &dims,
                         const std::string &voxel_type)
{
    Volume2D volume;
    volume.dims = dims;

    size_t voxel_size = 0;
    if (voxel_type == "uint8") {
        voxel_size = 1;
    } else if (voxel_type == "uint16") {
        voxel_size = 2;
    } else if (voxel_type == "float32") {
        voxel_size = 4;
    } else if (voxel_type == "float64") {
        voxel_size = 8;
    } else {
        throw std::runtime_error("Unrecognized voxel type " + voxel_type);
    }

    std::ifstream fin(fname.c_str(), std::ios::binary);
    auto voxel_data = std::make_shared<std::vector<uint8_t>>(volume.n_voxels() * voxel_size, 0);

    if (!fin.read(reinterpret_cast<char *>(voxel_data->data()), voxel_data->size())) {
        throw std::runtime_error("Failed to read volume " + fname);
    }

    volume.voxel_data = std::make_shared<std::vector<float>>(volume.n_voxels(), 0.f);

    // Temporarily convert non-float data to float
    // TODO will native support for non-float voxel types
    if (voxel_type == "uint8") {
        std::transform(voxel_data->begin(),
                       voxel_data->end(),
                       volume.voxel_data->begin(),
                       [](const uint8_t &x) { return float(x); });
    } else if (voxel_type == "uint16") {
        std::transform(reinterpret_cast<uint16_t *>(voxel_data->data()),
                       reinterpret_cast<uint16_t *>(voxel_data->data()) + volume.n_voxels(),
                       volume.voxel_data->begin(),
                       [](const uint16_t &x) { return float(x); });
    } else if (voxel_type == "float32") {
        std::transform(reinterpret_cast<float *>(voxel_data->data()),
                       reinterpret_cast<float *>(voxel_data->data()) + volume.n_voxels(),
                       volume.voxel_data->begin(),
                       [](const float &x) { return x; });
    } else {
        std::transform(reinterpret_cast<double *>(voxel_data->data()),
                       reinterpret_cast<double *>(voxel_data->data()) + volume.n_voxels(),
                       volume.voxel_data->begin(),
                       [](const double &x) { return float(x); });
    }
    
    // find the range
    volume.range.x = *std::min_element(volume.voxel_data->begin(), volume.voxel_data->end());
    volume.range.y = *std::max_element(volume.voxel_data->begin(), volume.voxel_data->end());
    // std::cout << "volume range: " << volume.range.x << " " << volume.range.y << std::endl;
    // convert to [-1, 1]
    // float old_range = volume.range.y - volume.range.x;
    // float new_range = 2.f;

    // float b = 1.f / (volume.range.y - volume.range.x) * 255.f;
    // std::vector<float> &voxels = *volume.voxel_data;
    // for(int i = 0; i < volume.n_voxels(); ++i){
    //     float a = (((voxels[i] - volume.range.x) * new_range) / old_range) + -1.f;
    //     voxels[i] = a;
    // }
    // volume.range.x = *std::min_element(volume.voxel_data->begin(), volume.voxel_data->end());
    // volume.range.y = *std::max_element(volume.voxel_data->begin(), volume.voxel_data->end());
    // std::cout << "volume new range: " << volume.range << std::endl;

 
    return volume;
}