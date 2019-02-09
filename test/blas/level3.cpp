#include <duda/blas/level3.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_gemm(
    const T alpha, const T beta, const int m, const int n, const int k)
{
    const auto a_d = duda::random_normal<T>(n, m);
    const auto b_d = duda::random_normal<T>(m, k);
    auto c_d       = duda::random_normal<T>(n, k);

    const auto a_h = testing::copy(a_d);
    const auto b_h = testing::copy(b_d);
    auto c_h       = testing::copy(c_d);

    gemm(duda::op::none, duda::op::none, alpha, a_d, b_d, beta, c_d);

    c_h = alpha * a_h * b_h + beta * c_h;

    REQUIRE(testing::all_close(c_d, c_h));
}

TEST_CASE("gemm", "[device_matrix][blas]")
{
    test_gemm<float>(0.1, 0.7, 16, 64, 16);
    test_gemm<double>(7, -0.7, 32, 16, 256);
}

template <typename T>
void test_gemm_transpose(
    const T alpha, const T beta, const int m, const int n, const int k)
{
    const auto a_d = duda::random_normal<T>(n, m);
    const auto b_d = duda::random_normal<T>(k, n);
    auto c_d       = duda::random_normal<T>(m, k);

    const auto a_h = testing::copy(a_d);
    const auto b_h = testing::copy(b_d);
    auto c_h       = testing::copy(c_d);

    duda::gemm(duda::op::transpose, duda::op::transpose, alpha, a_d, b_d, beta, c_d);

    c_h = alpha * a_h.transpose() * b_h.transpose() + beta * c_h;

    REQUIRE(testing::all_close(c_d, c_h));
}

TEST_CASE("gemm transpose ops", "[device_matrix][blas]")
{
    test_gemm_transpose<float>(0.1, 0.7, 512, 64, 8);
    test_gemm_transpose<double>(7, -0.7, 256, 8, 256);
}
