#include "../helpers.hpp"

#include "blas.hpp"
#include "random.hpp"

template <typename T>
void test_amax(const int n)
{
    auto x_d = duda::random_normal<T>(n);
    auto x_h = copy(x_d);

    int result;

    amax(x_d, result);

    typename host_vector<T>::Index index;

    x_h.cwiseAbs().maxCoeff(&index);

    REQUIRE(result == index);
}

TEST_CASE("amax", "[device_vector][blas]")
{
    test_amax<float>(512);
    test_amax<double>(16);
}

template <typename T>
void test_amin(const int n)
{
    auto x_d = duda::random_normal<T>(n);
    auto x_h = copy(x_d);

    int result;

    amin(x_d, result);

    typename host_vector<T>::Index index;

    x_h.cwiseAbs().minCoeff(&index);

    REQUIRE(result == index);
}

TEST_CASE("amin", "[device_vector][blas]")
{
    test_amin<float>(1080);
    test_amin<double>(666);
}

template <typename T, typename... Dim>
void test_asum(const Dim... dim)
{
    auto x_d = duda::random_normal<T>(dim...);
    auto x_h = copy(x_d);

    T result;

    asum(x_d, result);

    REQUIRE(result == Approx(x_h.cwiseAbs().sum()));
}

TEST_CASE("asum", "[device_vector][device_matrix][blas]")
{
    test_asum<float>(1080);
    test_asum<double>(666);

    test_asum<float>(32, 10);
    test_asum<double>(10, 10);
}

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

TEST_CASE("axpy", "[device_vector][device_matrix][blas]")
{
    test_axpy<float>(0.666, 196);
    test_axpy<double>(0.42, 1080);

    test_axpy<float>(3.14, 5, 6);
    test_axpy<double>(-3.14, 51, 16);
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
