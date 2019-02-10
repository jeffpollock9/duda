#include <duda/blas/level3.hpp>
#include <duda/random.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
constexpr T alpha = 0.001;

template <typename T>
constexpr T beta = 0.02;

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
static void BM_host_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const host_matrix<T> a = host_matrix<T>::Random(n, n);
    const host_matrix<T> b = host_matrix<T>::Random(n, n);
    host_matrix<T> c       = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        c = alpha<T> * a * b + beta<T> * c;
    }
}

template <typename T>
static void BM_device_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const auto a = duda::random_uniform<T>(n, n);
    const auto b = duda::random_uniform<T>(n, n);
    auto c       = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        duda::gemm(duda::op::none, duda::op::none, alpha<T>, a, b, beta<T>, c);
    }
}

#define DUDA_BENCHMARK_RANGE Range(8, 8 << 8)

BENCHMARK_TEMPLATE(BM_host_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
