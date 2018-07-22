#include "blas.hpp"
#include "device_matrix.hpp"

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

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

    device_matrix<T> x = device_matrix<T>::random_uniform(n, n);
    device_matrix<T> y = device_matrix<T>::random_uniform(n, n);

    for (auto _ : state)
    {
        axpy(a, x, y);
    }
}

#define RANGE Range(8, 8 << 8)

BENCHMARK_TEMPLATE(BM_host_axpy, float)->RANGE;
BENCHMARK_TEMPLATE(BM_host_axpy, double)->RANGE;

BENCHMARK_TEMPLATE(BM_device_axpy, float)->RANGE;
BENCHMARK_TEMPLATE(BM_device_axpy, double)->RANGE;

#undef RANGE

BENCHMARK_MAIN();
