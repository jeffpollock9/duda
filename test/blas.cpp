#include "helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

template <typename T, typename... Dim>
void test_axpy(const T alpha, const Dim... dim)
{
    auto x_d = duda::random_normal<T>(dim...);
    auto y_d = duda::random_normal<T>(dim...);

    auto x_h = copy(x_d);
    auto y_h = copy(y_d);

    axpy(alpha, x_d, y_d);
    y_h = alpha * x_h + y_h;

    REQUIRE(y_h.isApprox(copy(y_d)));
}

TEST_CASE("axpy matrix", "[device_matrix][blas]")
{
    test_axpy<float>(3.14, 5, 6);
    test_axpy<double>(-3.14, 51, 16);
}

TEST_CASE("axpy vector", "[device_vector][blas]")
{
    test_axpy<float>(3.14, 196);
    test_axpy<double>(-3.14, 1080);
}

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
void test_gemv(const T alpha, const T beta, const int m, const int n)
{
    auto A_d = duda::random_normal<T>(n, m);
    auto x_d = duda::random_normal<T>(m);
    auto y_d = duda::random_normal<T>(n);

    auto A_h = copy(A_d);
    auto x_h = copy(x_d);
    auto y_h = copy(y_d);

    gemv(duda::op::none, alpha, A_d, x_d, beta, y_d);

    y_h = alpha * A_h * x_h + beta * y_h;

    REQUIRE(y_h.isApprox(copy(y_d)));
}

TEST_CASE("gemv", "[device_matrix][device_vector][blas]")
{
    test_gemv<float>(0.1, 0.7, 16, 160);
    test_gemv<double>(7, -0.7, 16, 160);
}

template <typename T>
void test_gemv_transpose(const T alpha, const T beta, const int m, const int n)
{
    auto A_d = duda::random_normal<T>(n, m);
    auto x_d = duda::random_normal<T>(n);
    auto y_d = duda::random_normal<T>(m);

    auto A_h = copy(A_d);
    auto x_h = copy(x_d);
    auto y_h = copy(y_d);

    gemv(duda::op::transpose, alpha, A_d, x_d, beta, y_d);

    y_h = alpha * A_h.transpose() * x_h + beta * y_h;

    REQUIRE(y_h.isApprox(copy(y_d)));
}

TEST_CASE("gemv tranpose ops", "[device_matrix][device_vector][blas]")
{
    test_gemv<float>(0.1, 0.7, 16, 160);
    test_gemv<double>(7, -0.7, 16, 160);
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

template <typename T>
void test_dot(const int n)
{
    auto x_d = duda::random_normal<T>(n);
    auto y_d = duda::random_normal<T>(n);

    auto x_h = copy(x_d);
    auto y_h = copy(y_d);

    T result;

    dot(x_d, y_d, result);

    REQUIRE(result == Approx(x_h.cwiseProduct(y_h).sum()));
}

TEST_CASE("dot", "[device_vector][blas]")
{
    test_dot<double>(256);
    test_dot<double>(16);
}
