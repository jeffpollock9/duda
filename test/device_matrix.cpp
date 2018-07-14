#include "Eigen/Dense"
#include "catch/catch.hpp"

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

TEST_CASE("axpy", "[device_matrix]")
{
    const int rows = 3;
    const int cols = 4;

    SECTION("using float")
    {
        const float alpha = 3.14;

        auto x_d = device_matrix<float>::random(rows, cols);
        auto y_d = device_matrix<float>::random(rows, cols);

        host_matrix<float> x_h = copy(x_d);
        host_matrix<float> y_h = copy(y_d);

        duda::axpy(alpha, x_d, y_d);
        y_h += alpha * x_h;

        REQUIRE(y_h.isApprox(copy(y_d)));
    }

    SECTION("using double")
    {
        const double alpha = 42.0;

        auto x_d = device_matrix<double>::random(rows, cols);
        auto y_d = device_matrix<double>::random(rows, cols);

        host_matrix<double> x_h = copy(x_d);
        host_matrix<double> y_h = copy(y_d);

        duda::axpy(alpha, x_d, y_d);
        y_h += alpha * x_h;

        REQUIRE(y_h.isApprox(copy(y_d)));
    }
}
