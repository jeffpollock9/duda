#include <duda/device_vector.hpp>
#include <duda/reductions.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

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

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(100, 10'000'000)

#define DUDA_BENCHMARK_REDUCTION(reduction)                                    \
    template <typename T>                                                      \
    static void BM_host_##reduction(benchmark::State& state)                   \
    {                                                                          \
        std::srand(42);                                                        \
                                                                               \
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
        std::srand(42);                                                        \
                                                                               \
        const int n = state.range(0);                                          \
                                                                               \
        const host_vector<T> x_h = host_vector<T>::Random(n);                  \
        const duda::device_vector<T> x(x_h.data(), n);                         \
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

DUDA_BENCHMARK_REDUCTION(reduce_sum)
DUDA_BENCHMARK_REDUCTION(reduce_min)
DUDA_BENCHMARK_REDUCTION(reduce_max)
DUDA_BENCHMARK_REDUCTION(reduce_minmax)

int main(int argc, char** argv)
{
    rmmOptions_t options;
    options.allocation_mode   = rmmAllocationMode_t::PoolAllocation;
    options.initial_pool_size = 0;
    options.enable_logging    = false;

    const auto init = rmmInitialize(&options);
    std::cout << "init code: " << rmmGetErrorString(init) << "\n";

    ::benchmark::Initialize(&argc, argv);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();

    const auto fin_code = rmmFinalize();
    std::cout << "finalize code: " << rmmGetErrorString(fin_code) << "\n";
}
