#include <duda/random.hpp>
#include <duda/reductions/reduce_sum.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
static void BM_host_reduce_sum(benchmark::State& state)
{
    const int n = state.range(0);

    const auto x = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        const T sum = x.sum();

        benchmark::DoNotOptimize(sum);
    }
}

template <typename T>
static void BM_device_reduce_sum(benchmark::State& state)
{
    const int n = state.range(0);

    const auto x = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        const T sum = duda::reduce_sum(x);

        benchmark::DoNotOptimize(sum);
    }
}

#define DUDA_BENCHMARK_RANGE Range(8, 8 << 8)

BENCHMARK_TEMPLATE(BM_host_reduce_sum, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_reduce_sum, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_reduce_sum, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_reduce_sum, double)->DUDA_BENCHMARK_RANGE;

#undef DUDA_BENCHMARK_RANGE

BENCHMARK_MAIN();
