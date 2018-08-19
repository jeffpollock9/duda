find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(DUDA_BENCHMARKS axpy gemm reduce_sum)

foreach(benchmark ${DUDA_BENCHMARKS})
    add_executable(benchmark_${benchmark} ${benchmark}.cpp)
    target_link_libraries(benchmark_${benchmark}
        duda benchmark::benchmark Eigen3::Eigen OpenMP::OpenMP_CXX)
    # TODO: these flags are not portable
    target_compile_options(benchmark_${benchmark} PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow)
endforeach()
