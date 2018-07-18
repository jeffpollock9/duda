#include "Eigen/Dense"
#include "catch/catch.hpp"

#include "blas.hpp"
#include "device_matrix.hpp"

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

TEST_CASE("host -> device -> host", "[device_matrix]")
{
    const int rows = 4;
    const int cols = 4;

    SECTION("using float")
    {
        host_matrix<float> h1  = host_matrix<float>::Random(rows, cols);
        device_matrix<float> d = copy(h1);
        host_matrix<float> h2  = copy(d);

        REQUIRE(h1.isApprox(h2));
    }

    SECTION("using double")
    {
        host_matrix<double> h1  = host_matrix<double>::Random(rows, cols);
        device_matrix<double> d = copy(h1);
        host_matrix<double> h2  = copy(d);

        REQUIRE(h1.isApprox(h2));
    }
}

TEST_CASE("axpy", "[device_matrix][blas]")
{
    const int rows = 3;
    const int cols = 4;

    SECTION("using float")
    {
        const float alpha = 3.14;

        auto x_d = device_matrix<float>::random_normal(rows, cols);
        auto y_d = device_matrix<float>::random_normal(rows, cols);

        host_matrix<float> x_h = copy(x_d);
        host_matrix<float> y_h = copy(y_d);

        duda::axpy(alpha, x_d, y_d);
        y_h = alpha * x_h + y_h;

        REQUIRE(y_h.isApprox(copy(y_d)));
    }

    SECTION("using double")
    {
        const double alpha = 42.0;

        auto x_d = device_matrix<double>::random_normal(rows, cols);
        auto y_d = device_matrix<double>::random_normal(rows, cols);

        host_matrix<double> x_h = copy(x_d);
        host_matrix<double> y_h = copy(y_d);

        duda::axpy(alpha, x_d, y_d);
        y_h = alpha * x_h + y_h;

        REQUIRE(y_h.isApprox(copy(y_d)));
    }
}

TEST_CASE("gemm", "[device_matrix][blas]")
{
    const int rows = 4;
    const int cols = 4;

    SECTION("using float")
    {
        const float alpha = 3.14;
        const float beta  = 0.1;

        auto A_d = device_matrix<float>::random_normal(rows, cols);
        auto B_d = device_matrix<float>::random_normal(rows, cols);
        auto C_d = device_matrix<float>::random_normal(rows, cols);

        host_matrix<float> A_h = copy(A_d);
        host_matrix<float> B_h = copy(B_d);
        host_matrix<float> C_h = copy(C_d);

        duda::gemm(alpha, A_d, B_d, beta, C_d);
        C_h = alpha * A_h * B_h + beta * C_h;

        REQUIRE(C_h.isApprox(copy(C_d)));
    }

    SECTION("using double")
    {
        const double alpha = 7.0;
        const double beta  = -0.1;

        auto A_d = device_matrix<double>::random_normal(rows, cols);
        auto B_d = device_matrix<double>::random_normal(rows, cols);
        auto C_d = device_matrix<double>::random_normal(rows, cols);

        host_matrix<double> A_h = copy(A_d);
        host_matrix<double> B_h = copy(B_d);
        host_matrix<double> C_h = copy(C_d);

        duda::gemm(alpha, A_d, B_d, beta, C_d);
        C_h = alpha * A_h * B_h + beta * C_h;

        REQUIRE(C_h.isApprox(copy(C_d)));
    }
}
