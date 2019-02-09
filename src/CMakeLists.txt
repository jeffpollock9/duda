set(DUDA_KERNELS
    kernels/fill.cu
    kernels/eye.cu)

set(DUDA_REDUCTIONS
    reductions/reduce_sum.cu)

add_library(duda_kernels SHARED ${DUDA_KERNELS} ${DUDA_REDUCTIONS})

target_include_directories(duda_kernels PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_compile_features(duda_kernels PRIVATE cxx_std_14)
