#include <duda/blas.hpp>
#include <duda/random.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
constexpr T alpha = 0.001;

template <typename T>
constexpr T beta = 0.02;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
static void BM_host_gemv(benchmark::State& state)
{
    const int n = state.range(0);

    const auto a     = host_matrix<T>::Random(n, n);
    const auto x     = host_vector<T>::Random(n);
    host_vector<T> y = host_vector<T>::Random(n);

    for (auto _ : state)
    {
        y = alpha<T> * a * x + beta<T> * y;
    }
}

template <typename T>
static void BM_device_gemv(benchmark::State& state)
{
    const int n = state.range(0);

    const auto a = duda::random_uniform<T>(n, n);
    const auto x = duda::random_uniform<T>(n);
    auto y       = duda::random_uniform<T>(n);

    for (auto _ : state)
    {
        duda::gemv(duda::op::none, alpha<T>, a, x, beta<T>, y);
    }
}

#define DUDA_BENCHMARK_RANGE Range(8, 8 << 8)

BENCHMARK_TEMPLATE(BM_host_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemv, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemv, double)->DUDA_BENCHMARK_RANGE;

#undef DUDA_BENCHMARK_RANGE

BENCHMARK_MAIN();
