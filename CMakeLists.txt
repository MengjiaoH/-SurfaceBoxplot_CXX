cmake_minimum_required(VERSION 3.22)
project(SURFACE_BOXPLOT)

find_package(TBB REQUIRED)

add_library(util data_loader.cpp
                 helpers.cpp)

add_executable(main main.cpp)
target_link_libraries(main util TBB::tbb)

set_target_properties(main PROPERTIES CXX_STANDARD 11
                                      CXX_STANDARD_REQUIRED ON)