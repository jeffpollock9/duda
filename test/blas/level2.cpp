#include <helpers/helpers.hpp>

#include <duda/blas.hpp>
#include <duda/random.hpp>

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

TEST_CASE("gemv", "[device_vector][device_matrix][blas]")
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

TEST_CASE("gemv tranpose ops", "[device_vector][device_matrix][blas]")
{
    test_gemv<float>(0.1, 0.7, 16, 160);
    test_gemv<double>(7, -0.7, 16, 160);
}
