# Set project name
project(bitmatrix-multiplication)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

#Set sourses and binaries directory
set(SRC_LIST src/kernel.cu)

#Compile with c++11 possibilities
set(CMAKE_CXX_FLAGS "-std=c++11 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

# Specify binary name and source file to build it from
cuda_add_executable(../bin/${PROJECT_NAME} ${SRC_LIST})

# Specify library paths
include_directories(/home/libraries/)
