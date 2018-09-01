#include "../helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

template <typename T>
void test_gemm(
    const T alpha, const T beta, const int m, const int n, const int k)
{
    auto A_d = duda::random_normal<T>(n, m);
    auto B_d = duda::random_normal<T>(m, k);
    auto C_d = duda::random_normal<T>(n, k);

    auto A_h = copy(A_d);
    auto B_h = copy(B_d);
    auto C_h = copy(C_d);

    gemm(duda::op::none, duda::op::none, alpha, A_d, B_d, beta, C_d);

    C_h = alpha * A_h * B_h + beta * C_h;

    REQUIRE(C_h.isApprox(copy(C_d)));
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
    auto A_d = duda::random_normal<T>(n, m);
    auto B_d = duda::random_normal<T>(k, n);
    auto C_d = duda::random_normal<T>(m, k);

    auto A_h = copy(A_d);
    auto B_h = copy(B_d);
    auto C_h = copy(C_d);

    gemm(duda::op::transpose, duda::op::transpose, alpha, A_d, B_d, beta, C_d);

    C_h = alpha * A_h.transpose() * B_h.transpose() + beta * C_h;

    REQUIRE(C_h.isApprox(copy(C_d)));
}

TEST_CASE("gemm transpose ops", "[device_matrix][blas]")
{
    test_gemm_transpose<float>(0.1, 0.7, 512, 64, 8);
    test_gemm_transpose<double>(7, -0.7, 256, 8, 256);
}
