cmake_minimum_required(VERSION 3.24)

project(cuda_cute_from_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 方便编辑器读取编译参数。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 所有可执行文件统一输出到 build/bin。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(examples/cuda)
add_subdirectory(examples/cute)