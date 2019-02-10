#include <duda/random.hpp>
#include <duda/reductions.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(10, 1'000'000)

#define DUDA_BENCHMARK_REDUCTION(reduction)                                    \
    template <typename T>                                                      \
    static void BM_host_##reduction(benchmark::State& state)                   \
    {                                                                          \
        const int n = state.range(0);                                          \
                                                                               \
        const host_vector<T> x = host_vector<T>::Random(n);                    \
                                                                               \
        for (auto _ : state)                                                   \
        {                                                                      \
            const auto y = reduction(x);                                       \
                                                                               \
            benchmark::DoNotOptimize(y);                                       \
        }                                                                      \
    }                                                                          \
                                                                               \
    template <typename T>                                                      \
    static void BM_device_##reduction(benchmark::State& state)                 \
    {                                                                          \
        const int n = state.range(0);                                          \
                                                                               \
        const auto x = duda::random_uniform<T>(n);                             \
                                                                               \
        for (auto _ : state)                                                   \
        {                                                                      \
            const auto y = duda::reduction(x);                                 \
                                                                               \
            benchmark::DoNotOptimize(y);                                       \
        }                                                                      \
    }                                                                          \
                                                                               \
    BENCHMARK_TEMPLATE(BM_host_##reduction, float)->DUDA_BENCHMARK_RANGE;      \
    BENCHMARK_TEMPLATE(BM_host_##reduction, double)->DUDA_BENCHMARK_RANGE;     \
                                                                               \
    BENCHMARK_TEMPLATE(BM_device_##reduction, float)->DUDA_BENCHMARK_RANGE;    \
    BENCHMARK_TEMPLATE(BM_device_##reduction, double)->DUDA_BENCHMARK_RANGE;

template <typename T>
inline T reduce_sum(const host_vector<T>& x)
{
    return x.sum();
}

template <typename T>
inline T reduce_min(const host_vector<T>& x)
{
    return x.minCoeff();
}

template <typename T>
inline T reduce_max(const host_vector<T>& x)
{
    return x.maxCoeff();
}

template <typename T>
inline std::pair<T, T> reduce_minmax(const host_vector<T>& x)
{
    return {x.minCoeff(), x.maxCoeff()};
}

DUDA_BENCHMARK_REDUCTION(reduce_sum)
DUDA_BENCHMARK_REDUCTION(reduce_min)
DUDA_BENCHMARK_REDUCTION(reduce_max)
DUDA_BENCHMARK_REDUCTION(reduce_minmax)

BENCHMARK_MAIN();
