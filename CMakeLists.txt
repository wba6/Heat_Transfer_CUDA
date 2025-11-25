cmake_minimum_required(VERSION 4.1)
project(Heat_Transfer CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(${PROJECT_SOURCE_DIR}/vendor)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)



