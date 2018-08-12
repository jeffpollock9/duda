# duda TODO list

- manage dependencies via cmake(catch, google-benchmark, Eigen, cub)
- add cub, and add methods for reduction
- tidy up directories into modules, e.g. random, blas, kernels
- add all cuBLAS routines
- add all cuSOLVER routines
- add python bindings using pybind11
- add R bindings using Rcpp
- add useful numpy functions, like arange and linspace
- add matrix views
- allow (slow) access to device memory elements via operator(i, j)
