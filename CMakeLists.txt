cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g")

project(SLASH)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(lib/googletest)
