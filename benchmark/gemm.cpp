#include "helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

template <typename T>
static void BM_host_gemm(benchmark::State& state)
{
    const int n   = state.range(0);
    const T alpha = 0.001;
    const T beta  = 0.02;

    host_matrix<T> a = host_matrix<T>::Random(n, n);
    host_matrix<T> b = host_matrix<T>::Random(n, n);
    host_matrix<T> c = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        c = alpha * a * b + beta * c;
    }
}

template <typename T>
static void BM_device_gemm(benchmark::State& state)
{
    const int n   = state.range(0);
    const T alpha = 0.001;
    const T beta  = 0.02;

    device_matrix<T> a = duda::random_uniform<T>(n, n);
    device_matrix<T> b = duda::random_uniform<T>(n, n);
    device_matrix<T> c = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        gemm(duda::op::none, duda::op::none, alpha, a, b, beta, c);
    }
}

BENCHMARK_TEMPLATE(BM_host_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN()
