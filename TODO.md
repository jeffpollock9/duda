# duda TODO list

- add more cub reduction stuff + benchmarks
- tidy up directories into modules, e.g. random, blas, kernels
- add all cuBLAS routines
- add all cuSOLVER routines
- add statistical distributions
- add sparse matrix/vector
- add python bindings using pybind11
- add R bindings using Rcpp
- add useful numpy functions, like arange and linspace
- add matrix views
- allow (slow) access to device memory elements via operator(i, j)
- better logic for the number of threads per block and number of blocks, see
  e.g. https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
- see if we can use cub's CachingDeviceAllocator
- vector and matrix dimension thing
- allow mixing of matrix and vector in blas
- add CI on github with coverage etc
- fix project layout, see
  e.g. https://www.reddit.com/r/cpp/comments/996q8o/prepare_thy_pitchforks_a_de_facto_standard
- put helpers stuff into helpers namespace
- test complex number stuff
- use submodules for deps
