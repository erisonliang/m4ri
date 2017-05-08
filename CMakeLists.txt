# Set project name
project(m4ri)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

#Set sourses and binaries directory
set(SRC_LIST src/main.cu)

include_directories(src/x86, src/x64)

#Compile with c++11 possibilities
set(CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated-gpu-targets")
set(EXECUTABLE_OUTPUT_PATH "../bin/")

# Specify binary name and source file to build it from
cuda_add_executable(${PROJECT_NAME} ${SRC_LIST})
