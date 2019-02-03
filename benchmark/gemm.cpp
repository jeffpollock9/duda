#include <helpers/helpers.hpp>

#include <duda/blas.hpp>
#include <duda/random.hpp>

template <typename T>
constexpr T alpha = 0.001;

template <typename T>
constexpr T beta = 0.02;

template <typename T>
static void BM_host_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const host_matrix<T> A = host_matrix<T>::Random(n, n);
    const host_matrix<T> B = host_matrix<T>::Random(n, n);
    host_matrix<T> C       = host_matrix<T>::Random(n, n);

    for (auto _ : state)
    {
        C = alpha<T> * A * B + beta<T> * C;
    }
}

template <typename T>
static void BM_device_gemm(benchmark::State& state)
{
    const int n = state.range(0);

    const auto A = duda::random_uniform<T>(n, n);
    const auto B = duda::random_uniform<T>(n, n);
    auto C       = duda::random_uniform<T>(n, n);

    for (auto _ : state)
    {
        gemm(duda::op::none, duda::op::none, alpha<T>, A, B, beta<T>, C);
    }
}

BENCHMARK_TEMPLATE(BM_host_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_host_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_TEMPLATE(BM_device_gemm, float)->DUDA_BENCHMARK_RANGE;
BENCHMARK_TEMPLATE(BM_device_gemm, double)->DUDA_BENCHMARK_RANGE;

BENCHMARK_MAIN();
