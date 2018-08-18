#include "helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

template <typename T>
static void BM_host_axpy(benchmark::State& state)
{
    const int n = state.range(0);
    const T a   = 0.001;

    host_matrix<T> x = host_matrix<T>::Random(n, n);
    host_matrix<T> y = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        y = a * x + y;
    }
}

template <typename T>
static void BM_device_axpy(benchmark::State& state)
{
    const int n = state.range(0);
    const T a   = 0.001;

    device_matrix<T> x = duda::random_uniform<T>(n, n);
    device_matrix<T> y = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        axpy(a, x, y);
    }
}

BENCHMARK_TEMPLATE(BM_host_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN()
