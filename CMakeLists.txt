# default configuration
cmake_minimum_required(VERSION 3.10)
include(common.cmake OPTIONAL)

# project
project(nix CXX)

# package
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(Filesystem REQUIRED)

# build static library
add_library(nix STATIC balancer.cpp nixio.cpp sfc.cpp)

# include directory
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)

# unittest
if(ENABLE_TESTING)
	enable_testing()
	add_subdirectory(unittest)
endif()
