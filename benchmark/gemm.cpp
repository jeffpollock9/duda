#include <duda/blas/level3.hpp>

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
    std::srand(666);

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
    std::srand(666);

    const int n = state.range(0);

    const host_matrix<T> a_h = host_matrix<T>::Random(n, n);
    const host_matrix<T> b_h = host_matrix<T>::Random(n, n);
    const host_matrix<T> c_h = host_matrix<T>::Random(n, n);

    const duda::device_matrix<T> a(a_h.data(), n, n);
    const duda::device_matrix<T> b(b_h.data(), n, n);
    duda::device_matrix<T> c(c_h.data(), n, n);

    for (auto _ : state)
    {
        duda::gemm(duda::op::none, duda::op::none, alpha<T>, a, b, beta<T>, c);
    }
}

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(10, 1'000)

BENCHMARK_TEMPLATE(BM_host_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
