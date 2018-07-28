#include "helpers.hpp"

TEST_CASE("default ctor", "[device_matrix]")
{
    device_matrix<float> x;
    device_matrix<double> y;
}

template <typename T>
void test_copy_ctor(const int rows, const int cols)
{
    device_matrix<T> x_d = device_matrix<T>::random_normal(rows, cols);
    device_matrix<T> y_d(x_d);
    device_matrix<T> z_d = x_d;

    REQUIRE(x_d.data() != y_d.data());
    REQUIRE(x_d.data() != z_d.data());

    host_matrix<T> x_h = copy(x_d);
    host_matrix<T> y_h = copy(y_d);
    host_matrix<T> z_h = copy(z_d);

    REQUIRE(x_h.isApprox(y_h));
    REQUIRE(x_h.isApprox(z_h));
}

TEST_CASE("copy ctor", "[device_matrix]")
{
    test_copy_ctor<float>(16, 16);
    test_copy_ctor<double>(32, 32);
}

template <typename T>
void test_move_ctor(const int rows, const int cols)
{
    device_matrix<T> x_d = device_matrix<T>::random_normal(rows, cols);
    host_matrix<T> x_h   = copy(x_d);

    device_matrix<T> y_d = std::move(x_d);
    host_matrix<T> y_h   = copy(y_d);

    REQUIRE(x_d.data() == nullptr);

    REQUIRE(x_h.isApprox(y_h));
    REQUIRE(y_d.rows() == rows);
    REQUIRE(y_d.cols() == cols);
}

TEST_CASE("move ctor", "[device_matrix]")
{
    test_move_ctor<float>(16, 16);
    test_move_ctor<double>(32, 32);
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
    test_transfer<float>(4, 5);
    test_transfer<double>(10, 1);
}
