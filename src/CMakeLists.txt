set(DUDA_MATH
    math/exp.cu
    math/log.cu
    )

set(DUDA_KERNELS
    kernels/fill.cu
    kernels/eye.cu
    )

set(DUDA_REDUCTIONS
    reductions/reduce_sum.cu
    reductions/reduce_min.cu
    reductions/reduce_max.cu
    reductions/reduce_minmax.cu
    )

add_library(duda_kernels SHARED ${DUDA_MATH} ${DUDA_KERNELS} ${DUDA_REDUCTIONS})

target_include_directories(duda_kernels PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(duda_kernels PRIVATE rmm::rmm)

target_compile_features(duda_kernels PRIVATE cxx_std_14)
