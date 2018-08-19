cmake_minimum_required(VERSION 3.9)

project(duda LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

option(DUDA_TEST "Build tests" OFF)
option(DUDA_BENCHMARK "Build benchmarks" OFF)
option(DUDA_EXAMPLE "Build examples" OFF)

message(STATUS "Configuring ${PROJECT_NAME} version: ${PROJECT_VERSION}")
message(STATUS "  Options:")
message(STATUS "   DUDA_TEST: ${DUDA_TEST}")
message(STATUS "   DUDA_BENCHMARK: ${DUDA_BENCHMARK}")
message(STATUS "   DUDA_EXAMPLE: ${DUDA_EXAMPLE}")

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src)

add_library(duda INTERFACE)

target_link_libraries(duda INTERFACE
    duda_kernels
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY})

target_include_directories(duda INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CUDA_TOOLKIT_INCLUDE})

target_compile_features(duda INTERFACE cxx_std_14)

if(DUDA_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

if(DUDA_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(DUDA_EXAMPLE)
    add_subdirectory(example)
endif()
