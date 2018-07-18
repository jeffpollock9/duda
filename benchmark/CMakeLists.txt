find_package(benchmark)
find_package(Eigen3)
find_package(OpenMP)

set(DUDA_BENCHMARKS axpy gemm)

foreach(benchmark ${DUDA_BENCHMARKS})
    add_executable(${benchmark} ${benchmark}.cpp)
    target_link_libraries(${benchmark} duda benchmark::benchmark Eigen3::Eigen OpenMP::OpenMP_CXX)
    # TODO: these flags are not portable
    target_compile_options(${benchmark} PUBLIC -Wall -Wextra -Wpedantic -Wshadow -march=native)
endforeach()
