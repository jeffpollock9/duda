# duda TODO list

- manage dependencies via cmake(catch, google-benchmark, Eigen, cub)
- add more cub reduction stuff + benchmarks
- tidy up directories into modules, e.g. random, blas, kernels
- add all cuBLAS routines
- add all cuSOLVER routines
- add python bindings using pybind11
- add R bindings using Rcpp
- add useful numpy functions, like arange and linspace
- add matrix views
- allow (slow) access to device memory elements via operator(i, j)
- better logic for the number of threads per block and number of blocks, see
  e.g. https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
- see if we can use cub's CachingDeviceAllocator
