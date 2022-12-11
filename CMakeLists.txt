cmake_minimum_required(VERSION 3.22)
project(SURFACE_BOXPLOT)

add_library(util data_loader.cpp
                 helpers.cpp)

add_executable(main main.cpp)
target_link_libraries(main util)

set_target_properties(main PROPERTIES CXX_STANDARD 11
                                      CXX_STANDARD_REQUIRED ON)