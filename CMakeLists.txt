# default configuration
cmake_minimum_required(VERSION 3.10)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common.cmake OPTIONAL)

# project
project(libnix CXX)

# include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

# build static library
add_library(nix STATIC balancer.cpp jsonio.cpp sfc.cpp)

# to be included
target_include_directories(nix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
