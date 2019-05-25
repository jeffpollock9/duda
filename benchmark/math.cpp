#include <duda/device_vector.hpp>
#include <duda/math.hpp>
#include <duda/memory_manager.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
using host_vector = Eigen::Array<T, Eigen::Dynamic, 1>;

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(100, 10'000'000)

#define DUDA_BENCHMARK_MATH(function)                                          \
    template <typename T>                                                      \
    static void BM_host_##function(benchmark::State& state)                    \
    {                                                                          \
        std::srand(666);                                                       \
                                                                               \
        const int n = state.range(0);                                          \
                                                                               \
        const host_vector<T> x = host_vector<T>::Random(n) + 1.0;              \
                                                                               \
        for (auto _ : state)                                                   \
        {                                                                      \
            const host_vector<T> y = x.function();                             \
                                                                               \
            benchmark::DoNotOptimize(y);                                       \
        }                                                                      \
    }                                                                          \
                                                                               \
    template <typename T>                                                      \
    static void BM_device_##function(benchmark::State& state)                  \
    {                                                                          \
        std::srand(666);                                                       \
                                                                               \
        const int n = state.range(0);                                          \
                                                                               \
        const host_vector<T> x_h = host_vector<T>::Random(n) + 1.0;            \
        const duda::device_vector<T> x(x_h.data(), n);                         \
                                                                               \
        for (auto _ : state)                                                   \
        {                                                                      \
            const auto y = duda::function(x);                                  \
                                                                               \
            benchmark::DoNotOptimize(y);                                       \
        }                                                                      \
    }                                                                          \
                                                                               \
    BENCHMARK_TEMPLATE(BM_host_##function, float)->DUDA_BENCHMARK_RANGE;       \
    BENCHMARK_TEMPLATE(BM_host_##function, double)->DUDA_BENCHMARK_RANGE;      \
                                                                               \
    BENCHMARK_TEMPLATE(BM_device_##function, float)->DUDA_BENCHMARK_RANGE;     \
    BENCHMARK_TEMPLATE(BM_device_##function, double)->DUDA_BENCHMARK_RANGE;

DUDA_BENCHMARK_MATH(exp)
DUDA_BENCHMARK_MATH(log)

int main(int argc, char** argv)
{
    duda::memory_manager memory_manager;

    benchmark::Initialize(&argc, argv);

    if (benchmark::ReportUnrecognizedArguments(argc, argv))
    {
        return 1;
    }

    benchmark::RunSpecifiedBenchmarks();
}
