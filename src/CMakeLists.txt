find_package(cub REQUIRED)

add_library(duda_kernels SHARED eye.cu fill.cu reduce_sum.cu)

target_link_libraries(duda_kernels PRIVATE cub::cub)

target_include_directories(duda_kernels PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_compile_features(duda_kernels PRIVATE cxx_std_14)
