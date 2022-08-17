cmake_minimum_required (VERSION 3.9)
project(pytorch_cmake_example LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 60 75 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)

include_directories(extern/include
                    extern)

add_library(kernel SHARED extern/kernel.cu)
add_executable(debug_main extern/test.cu)
target_link_libraries(debug_main PRIVATE kernel)
