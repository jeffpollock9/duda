#include <duda/device_matrix.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

TEST_CASE("default ctor", "[device_matrix]")
{
    duda::device_matrix<float> x;
    duda::device_matrix<double> y;
}

template <typename T>
void test_copy_ctor(const int rows, const int cols)
{
    duda::device_matrix<T> x_d = duda::random_normal<T>(rows, cols);
    duda::device_matrix<T> y_d(x_d);
    duda::device_matrix<T> z_d = x_d;

    REQUIRE(x_d.data() != y_d.data());
    REQUIRE(x_d.data() != z_d.data());

    testing::host_matrix<T> x_h = testing::copy(x_d);
    testing::host_matrix<T> y_h = testing::copy(y_d);
    testing::host_matrix<T> z_h = testing::copy(z_d);

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
    duda::device_matrix<T> x_d  = duda::random_normal<T>(rows, cols);
    testing::host_matrix<T> x_h = testing::copy(x_d);

    duda::device_matrix<T> y_d  = std::move(x_d);
    testing::host_matrix<T> y_h = testing::copy(y_d);

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
    testing::host_matrix<T> h1 = testing::host_matrix<T>::Random(rows, cols);
    duda::device_matrix<T> d   = testing::copy(h1);
    testing::host_matrix<T> h2 = testing::copy(d);

    REQUIRE(h1.isApprox(h2));
}

TEST_CASE("host -> device -> host", "[device_matrix]")
{
    test_transfer<float>(4, 5);
    test_transfer<double>(10, 1);
}
