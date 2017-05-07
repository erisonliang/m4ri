project(m4ri)

set(SRC_LIST main.cu)

include_directories(x86)
include_directories(x64)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(EXECUTABLE_OUTPUT_PATH "../../bin/")

cuda_add_executable(${PROJECT_NAME} ${SRC_LIST})