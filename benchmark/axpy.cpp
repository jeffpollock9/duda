#include <duda/blas/level1.hpp>
#include <duda/random.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
constexpr T a = 0.001;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
static void BM_host_axpy(benchmark::State& state)
{
    const int n = state.range(0);

    const host_vector<T> x = host_vector<T>::Random(n);
    host_vector<T> y       = host_vector<T>::Random(n);

    for (auto _ : state)
    {
        y = a<T> * x + y;
    }
}

template <typename T>
static void BM_device_axpy(benchmark::State& state)
{
    const int n = state.range(0);

    const auto x = duda::random_uniform<T>(n);
    auto y       = duda::random_uniform<T>(n);

    for (auto _ : state)
    {
        axpy(a<T>, x, y);
    }
}

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(10, 1'000'000)

BENCHMARK_TEMPLATE(BM_host_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
