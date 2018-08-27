#include "helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

template <typename T>
constexpr T alpha = 0.001;

template <typename T>
constexpr T beta = 0.02;

template <typename T>
static void BM_host_gemv(benchmark::State& state)
{
    const int n = state.range(0);

    const host_matrix<T> A = host_matrix<T>::Random(n, n);
    const host_vector<T> x = host_vector<T>::Random(n);
    host_vector<T> y       = host_vector<T>::Random(n);

    for (auto _ : state)
    {
        y = alpha<T> * A * x + beta<T> * y;
    }
}

template <typename T>
static void BM_host_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const host_matrix<T> A = host_matrix<T>::Random(n, n);
    const host_matrix<T> x = host_vector<T>::Random(n, 1);
    host_matrix<T> y       = host_vector<T>::Random(n, 1);

    for (auto _ : state)
    {
        y = alpha<T> * A * x + beta<T> * y;
    }
}

template <typename T>
static void BM_device_gemv(benchmark::State& state)
{
    const int n = state.range(0);

    const auto A = duda::random_uniform<T>(n, n);
    const auto x = duda::random_uniform<T>(n);
    auto y       = duda::random_uniform<T>(n);

    for (auto _ : state)
    {
        gemv(duda::op::none, alpha<T>, A, x, beta<T>, y);
    }
}

template <typename T>
static void BM_device_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const auto A = duda::random_uniform<T>(n, n);
    const auto x = duda::random_uniform<T>(n, 1);
    auto y       = duda::random_uniform<T>(n, 1);

    for (auto _ : state)
    {
        gemm(duda::op::none, duda::op::none, alpha<T>, A, x, beta<T>, y);
    }
}

BENCHMARK_TEMPLATE(BM_host_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemv, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_host_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemv, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN()
