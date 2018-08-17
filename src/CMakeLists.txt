add_library(duda_kernels SHARED eye.cu fill.cu reduce_sum.cu)

target_compile_features(duda_kernels PRIVATE cxx_std_14)
