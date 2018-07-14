find_package(benchmark)
find_package(Eigen3)

set(DUDA_BENCHMARKS axpy)

foreach(benchmark ${DUDA_BENCHMARKS})
    add_executable(${benchmark} ${benchmark}.cpp)
    target_link_libraries(${benchmark} duda benchmark::benchmark Eigen3::Eigen)
    # TODO: these flags are not portable
    target_compile_options(${benchmark} PUBLIC -Wall -Wextra -Wpedantic -Wshadow -march=native)
endforeach()
