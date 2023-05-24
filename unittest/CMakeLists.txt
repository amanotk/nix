# default configuration
cmake_minimum_required(VERSION 3.10)
include(../common.cmake OPTIONAL)

# project
set(PROJECT_NAME "unittest")
project(${PROJECT_NAME} CXX)

set(INCLUDE_DIRS
	${CMAKE_BINARY_DIR}/nix
	${CMAKE_BINARY_DIR}/nix/thirdparty
	)
set(LIBRARY_DIRS
	${CMAKE_BINARY_DIR}/nix
	)

#
# serial testing
#
set(SERIAL_EXECS
	test_balancer
	test_sfc
	test_nixio
	test_chunk
	test_chunkmap
	test_xtensor_particle
	)

add_library(obj_serial OBJECT test_serial.cpp)
target_include_directories(obj_serial PRIVATE ${INCLUDE_DIRS})
target_link_directories(obj_serial PRIVATE ${LIBRARY_DIRS})
target_link_libraries(obj_serial nix)

foreach(target IN ITEMS ${SERIAL_EXECS})
	add_executable(${target} ${target}.cpp)
	target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
	target_link_directories(${target} PRIVATE ${LIBRARY_DIRS})
	target_link_libraries(${target} obj_serial)
	target_link_libraries(${target} nix)
	add_test(NAME ${target} COMMAND ${target})
endforeach()

#
# parallel testing
#
set(PARALLEL_EXECS
	test_nixio_parallel
	)

add_library(obj_parallel OBJECT test_parallel.cpp)
target_include_directories(obj_parallel PRIVATE ${INCLUDE_DIRS})
target_link_directories(obj_parallel PRIVATE ${LIBRARY_DIRS})
target_link_libraries(obj_parallel nix)

foreach(target IN ITEMS ${PARALLEL_EXECS})
	add_executable(${target} ${target}.cpp)
	target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
	target_link_directories(${target} PRIVATE ${LIBRARY_DIRS})
	target_link_libraries(${target} obj_parallel)
	target_link_libraries(${target} nix)
endforeach()

add_test(NAME test_nixio_parallel
	COMMAND mpiexec -n 8 ./test_nixio_parallel -X 2 -Y 2 -Z 2
	)