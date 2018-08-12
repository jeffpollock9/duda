#include "helpers.hpp"

template <typename T>
void test_axpy(const T alpha, const int rows, const int cols)
{
    auto x_d = duda::random_normal<T>(rows, cols);
    auto y_d = duda::random_normal<T>(rows, cols);

    host_matrix<T> x_h = copy(x_d);
    host_matrix<T> y_h = copy(y_d);

    axpy(alpha, x_d, y_d);
    y_h = alpha * x_h + y_h;

    REQUIRE(y_h.isApprox(copy(y_d)));
}

TEST_CASE("axpy", "[device_matrix][blas]")
{
    test_axpy<float>(3.14, 5, 6);
    test_axpy<double>(-3.14, 51, 16);
}

template <typename T>
void test_gemm(
    const T alpha, const T beta, const int m, const int n, const int k)
{
    auto A_d = duda::random_normal<T>(n, m);
    auto B_d = duda::random_normal<T>(m, k);
    auto C_d = duda::random_normal<T>(n, k);

    host_matrix<T> A_h = copy(A_d);
    host_matrix<T> B_h = copy(B_d);
    host_matrix<T> C_h = copy(C_d);

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

    host_matrix<T> A_h = copy(A_d);
    host_matrix<T> B_h = copy(B_d);
    host_matrix<T> C_h = copy(C_d);

    gemm(duda::op::transpose, duda::op::transpose, alpha, A_d, B_d, beta, C_d);

    C_h = alpha * A_h.transpose() * B_h.transpose() + beta * C_h;

    REQUIRE(C_h.isApprox(copy(C_d)));
}

TEST_CASE("gemm transpose ops", "[device_matrix][blas]")
{
    test_gemm_transpose<float>(0.1, 0.7, 512, 64, 8);
    test_gemm_transpose<double>(7, -0.7, 256, 8, 256);
}

template <typename T>
void test_dot(const int n)
{
    auto x_d = duda::random_normal<T>(n, 1);
    auto y_d = duda::random_normal<T>(n, 1);

    host_matrix<T> x_h = copy(x_d);
    host_matrix<T> y_h = copy(y_d);

    T result;

    dot(x_d, y_d, result);

    REQUIRE(result == Approx(x_h.cwiseProduct(y_h).sum()));
}

TEST_CASE("dot", "[device_matrix][blas]")
{
    test_dot<double>(256);
    test_dot<double>(16);
}
