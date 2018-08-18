#include "helpers.hpp"

#include "random.hpp"
#include "reduce_sum.hpp"

template <typename T>
static void BM_host_reduce_sum(benchmark::State& state)
{
    const int n = state.range(0);

    host_matrix<T> x = host_matrix<T>::Random(n, n);

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

    device_matrix<T> x = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        const T sum = reduce_sum(x);

        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK_TEMPLATE(BM_host_reduce_sum, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_reduce_sum, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_reduce_sum, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_reduce_sum, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN()
