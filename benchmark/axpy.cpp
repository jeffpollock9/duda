#include <duda/blas/level1.hpp>
#include <duda/memory_manager.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
constexpr T a = 0.001;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
static void BM_host_axpy(benchmark::State& state)
{
    std::srand(123);

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
    std::srand(123);

    const int n = state.range(0);

    const host_vector<T> x_h = host_vector<T>::Random(n);
    const host_vector<T> y_h = host_vector<T>::Random(n);

    const duda::device_vector<T> x(x_h.data(), n);
    duda::device_vector<T> y(y_h.data(), n);

    for (auto _ : state)
    {
        duda::axpy(a<T>, x, y);
    }
}

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(100, 10'000'000)

BENCHMARK_TEMPLATE(BM_host_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_axpy, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_axpy, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_axpy, double)->DUDA_BENCHMARK_RANGE;

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
