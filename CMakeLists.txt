# default configuration
cmake_minimum_required(VERSION 3.10)
include(common.cmake OPTIONAL)

# project
project(nix CXX)

# build static library
add_library(nix STATIC balancer.cpp jsonio.cpp sfc.cpp)

# include directory
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)
