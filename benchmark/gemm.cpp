#include "Eigen/Dense"
#include "benchmark/benchmark.h"

#include "blas.hpp"
#include "device_matrix.hpp"

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

template <typename T>
static void BM_host_gemm(benchmark::State& state)
{
    const int n   = state.range(0);
    const T alpha = 0.001;
    const T beta  = 0.02;

    host_matrix<T> A = host_matrix<T>::Random(n, n);
    host_matrix<T> B = host_matrix<T>::Random(n, n);
    host_matrix<T> C = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        C = alpha * A * B + beta * C;
    }
}

template <typename T>
static void BM_device_gemm(benchmark::State& state)
{
    const int n   = state.range(0);
    const T alpha = 0.001;
    const T beta  = 0.02;

    device_matrix<T> A = device_matrix<T>::random_uniform(n, n);
    device_matrix<T> B = device_matrix<T>::random_uniform(n, n);
    device_matrix<T> C = device_matrix<T>::random_uniform(n, n);

    for (auto _ : state)
    {
        gemm(alpha, A, B, beta, C);
    }
}

#define RANGE Range(8, 8 << 8)

BENCHMARK_TEMPLATE(BM_host_gemm, float)->RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->RANGE;

#undef RANGE

BENCHMARK_MAIN();
