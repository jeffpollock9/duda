include(add_basic_warnings)

find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(DUDA_BENCHMARKS axpy gemm gemv reduce_sum)

foreach(benchmark ${DUDA_BENCHMARKS})
    add_executable(benchmark_${benchmark} ${benchmark}.cpp)
    target_link_libraries(benchmark_${benchmark}
        duda benchmark::benchmark Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_include_directories(benchmark_${benchmark} PRIVATE ${PROJECT_SOURCE_DIR}/benchmark)
    add_basic_warnings(benchmark_${benchmark})
endforeach()
