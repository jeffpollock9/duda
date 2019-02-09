#include <duda/blas/level2.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_gemv(const T alpha, const T beta, const int m, const int n)
{
    const auto a_d = duda::random_normal<T>(n, m);
    const auto x_d = duda::random_normal<T>(m);
    auto y_d = duda::random_normal<T>(n);

    const auto a_h = testing::copy(a_d);
    const auto x_h = testing::copy(x_d);
    auto y_h = testing::copy(y_d);

    duda::gemv(duda::op::none, alpha, a_d, x_d, beta, y_d);

    y_h = alpha * a_h * x_h + beta * y_h;

    REQUIRE(testing::all_close(y_d, y_h));
}

TEST_CASE("gemv", "[device_vector][device_matrix][blas]")
{
    test_gemv<float>(0.1, 0.7, 16, 160);
    test_gemv<double>(7, -0.7, 16, 160);
}

template <typename T>
void test_gemv_transpose(const T alpha, const T beta, const int m, const int n)
{
    const auto a_d = duda::random_normal<T>(n, m);
    const auto x_d = duda::random_normal<T>(n);
    auto y_d = duda::random_normal<T>(m);

    const auto a_h = testing::copy(a_d);
    const auto x_h = testing::copy(x_d);
    auto y_h = testing::copy(y_d);

    duda::gemv(duda::op::transpose, alpha, a_d, x_d, beta, y_d);

    y_h = alpha * a_h.transpose() * x_h + beta * y_h;

    REQUIRE(testing::all_close(y_d, y_h));
}

TEST_CASE("gemv tranpose ops", "[device_vector][device_matrix][blas]")
{
    test_gemv<float>(0.1, 0.7, 16, 160);
    test_gemv<double>(7, -0.7, 16, 160);
}

template <typename T>
void test_syr(const T alpha, const int n)
{
    const auto x_d = duda::random_normal<T>(n);
    auto a_d = duda::random_normal<T>(n, n);

    const auto x_h = testing::copy(x_d);
    auto a_h = testing::copy(a_d);

    duda::syr(duda::fill_mode::lower, alpha, x_d, a_d);

    a_h = alpha * x_h * x_h.transpose() + a_h;

    testing::host_matrix<T> a_d_h = testing::copy(a_d);

    for (int j = 0; j < n; ++j)
    {
        for (int i = j; i < n; ++i)
        {
            REQUIRE(a_h(i, j) == Approx(a_d_h(i, j)));
        }
    }
}

TEST_CASE("syr", "[device_vector][device_matrix]")
{
    test_syr<float>(3.14, 24);
    test_syr<float>(-0.42, 32);
}
