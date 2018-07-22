#include "device_matrix.hpp"
#include "blas.hpp"

#include "Eigen/Dense"
#include "catch/catch.hpp"

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

template <typename T>
host_matrix<T> copy(const device_matrix<T>& device)
{
    host_matrix<T> host(device.rows(), device.cols());

    duda::copy(device, host.data());

    return host;
}

template <typename T>
device_matrix<T> copy(const host_matrix<T>& host)
{
    const int rows = host.rows();
    const int cols = host.cols();

    return {host.data(), rows, cols};
}

TEST_CASE("default_ctor", "[device_matrix]")
{
    SECTION("using float")
    {
        device_matrix<float> x;
    }

    SECTION("using double")
    {
        device_matrix<double> x;
    }
}

template <typename T>
void test_copy_ctor(const int rows, const int cols)
{
    device_matrix<T> x_d = device_matrix<T>::random_normal(rows, cols);
    device_matrix<T> y_d(x_d);

    REQUIRE(x_d.data() != y_d.data());

    host_matrix<T> x_h = copy(x_d);
    host_matrix<T> y_h = copy(y_d);

    REQUIRE(x_h.isApprox(y_h));
}

TEST_CASE("copy_ctor", "[device_matrix]")
{
    SECTION("using float")
    {
        test_copy_ctor<float>(16, 16);
    }

    SECTION("using double")
    {
        test_copy_ctor<double>(32, 32);
    }
}

template <typename T>
void test_transfer(const int rows, const int cols)
{
    host_matrix<T> h1  = host_matrix<T>::Random(rows, cols);
    device_matrix<T> d = copy(h1);
    host_matrix<T> h2  = copy(d);

    REQUIRE(h1.isApprox(h2));
}

TEST_CASE("host -> device -> host", "[device_matrix]")
{
    SECTION("using float")
    {
        test_transfer<float>(4, 5);
    }

    SECTION("using double")
    {
        test_transfer<double>(10, 1);
    }
}

template <typename T>
void test_axpy(const T alpha, const int rows, const int cols)
{
    auto x_d = device_matrix<T>::random_normal(rows, cols);
    auto y_d = device_matrix<T>::random_normal(rows, cols);

    host_matrix<T> x_h = copy(x_d);
    host_matrix<T> y_h = copy(y_d);

    duda::axpy(alpha, x_d, y_d);
    y_h = alpha * x_h + y_h;

    REQUIRE(y_h.isApprox(copy(y_d)));
}

TEST_CASE("axpy", "[device_matrix][blas]")
{
    SECTION("using float")
    {
        test_axpy<float>(3.14, 5, 6);
    }

    SECTION("using double")
    {
        test_axpy<double>(-3.14, 51, 16);
    }
}

template <typename T>
void test_gemm(const T alpha, const T beta, const int rows, const int cols)
{
    auto A_d = device_matrix<T>::random_normal(rows, cols);
    auto B_d = device_matrix<T>::random_normal(rows, cols);
    auto C_d = device_matrix<T>::random_normal(rows, cols);

    host_matrix<T> A_h = copy(A_d);
    host_matrix<T> B_h = copy(B_d);
    host_matrix<T> C_h = copy(C_d);

    duda::gemm(duda::op::none, duda::op::none, alpha, A_d, B_d, beta, C_d);
    C_h = alpha * A_h * B_h + beta * C_h;

    REQUIRE(C_h.isApprox(copy(C_d)));
}

TEST_CASE("gemm", "[device_matrix][blas]")
{
    SECTION("using float")
    {
        test_gemm<float>(0.1, 0.7, 10, 10);
    }

    SECTION("using double")
    {
        test_gemm<double>(7, -0.7, 100, 2);
    }
}
