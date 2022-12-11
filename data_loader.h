#pragma once 

#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>

struct vec2i {
    int x = 0;
    int y = 0;
    vec2i(int xx, int yy){
        x = xx;
        y = yy;
    }
    vec2i(int xx){
        x = xx;
        y = xx;
    }
};

struct vec2f {
    float x = 0;
    float y = 0;
    vec2f(float xx, float yy){
        x = xx;
        y = yy;
    }
    vec2f(float xx){
        x = xx;
        y = xx;
    }
};

struct vec3f {
    float x = 0;
    float y = 0;
    float z = 0;
    vec3f(float xx, float yy, float zz){
        x = xx;
        y = yy;
        z = zz;
    }
    vec3f(float xx){
        x = xx;
        y = xx;
        z = xx;
    }
};


struct Volume2D {
    vec2i dims = vec2i(0);
    vec2f range = vec2f(0);
    vec3f spacing{1.f};
    vec3f origin{0.f};
    int timestep = 0;
    std::shared_ptr<std::vector<float>> voxel_data = nullptr;

    size_t n_voxels() const
    {
        return size_t(dims.x) * size_t(dims.y);
    }
};

Volume2D load_raw_volume(const std::string &fname,
                         const vec2i &dims,
                         const std::string &voxel_type);