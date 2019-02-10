#include <duda/blas/level2.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <typename T>
constexpr T alpha = 0.001;

template <typename T>
constexpr T beta = 0.02;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
static void BM_host_gemv(benchmark::State& state)
{
    std::srand(42);

    const int n = state.range(0);

    const host_matrix<T> a = host_matrix<T>::Random(n, n);
    const host_vector<T> x = host_vector<T>::Random(n);
    host_vector<T> y       = host_vector<T>::Random(n);

    for (auto _ : state)
    {
        y = alpha<T> * a * x + beta<T> * y;
    }
}

template <typename T>
static void BM_device_gemv(benchmark::State& state)
{
    std::srand(42);

    const int n = state.range(0);

    const host_matrix<T> a_h = host_matrix<T>::Random(n, n);
    const host_vector<T> x_h = host_vector<T>::Random(n);
    const host_vector<T> y_h = host_vector<T>::Random(n);

    const duda::device_matrix<T> a(a_h.data(), n, n);
    const duda::device_vector<T> x(x_h.data(), n);
    duda::device_vector<T> y(y_h.data(), n);

    for (auto _ : state)
    {
        duda::gemv(duda::op::none, alpha<T>, a, x, beta<T>, y);
    }
}

#define DUDA_BENCHMARK_RANGE RangeMultiplier(10)->Range(10, 1'000)

BENCHMARK_TEMPLATE(BM_host_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemv, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemv, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemv, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
