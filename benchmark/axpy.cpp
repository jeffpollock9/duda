#include <helpers/helpers.hpp>

#include <duda/blas.hpp>
#include <duda/random.hpp>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

template <typename T>
constexpr T a = 0.001;

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

BENCHMARK_TEMPLATE(BM_host_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
