#include <duda/random.hpp>
#include <duda/device_vector.hpp>

#include <testing.hpp>

TEST_CASE("default ctor", "[device_vector]")
{
    duda::device_vector<float> x;
    duda::device_vector<double> y;
}

template <typename T>
void test_copy_ctor(const int size)
{
    duda::device_vector<T> x_d = duda::random_normal<T>(size);
    duda::device_vector<T> y_d(x_d);
    duda::device_vector<T> z_d = x_d;

    REQUIRE(x_d.data() != y_d.data());
    REQUIRE(x_d.data() != z_d.data());

    testing::host_vector<T> x_h = testing::copy(x_d);
    testing::host_vector<T> y_h = testing::copy(y_d);
    testing::host_vector<T> z_h = testing::copy(z_d);

    REQUIRE(x_h.isApprox(y_h));
    REQUIRE(x_h.isApprox(z_h));
}

TEST_CASE("copy ctor", "[device_vector]")
{
    test_copy_ctor<float>(1080);
    test_copy_ctor<double>(256);
}

template <typename T>
void test_move_ctor(const int size)
{
    duda::device_vector<T> x_d  = duda::random_normal<T>(size);
    testing::host_vector<T> x_h = testing::copy(x_d);

    duda::device_vector<T> y_d  = std::move(x_d);
    testing::host_vector<T> y_h = testing::copy(y_d);

    REQUIRE(x_d.data() == nullptr);

    REQUIRE(x_h.isApprox(y_h));
    REQUIRE(y_d.size() == size);
}

TEST_CASE("move ctor", "[device_vector]")
{
    test_move_ctor<float>(16);
    test_move_ctor<double>(512);
}

template <typename T>
void test_transfer(const int size)
{
    testing::host_vector<T> h1 = testing::host_vector<T>::Random(size);
    duda::device_vector<T> d   = testing::copy(h1);
    testing::host_vector<T> h2 = testing::copy(d);

    REQUIRE(h1.isApprox(h2));
}

TEST_CASE("host -> device -> host", "[device_vector]")
{
    test_transfer<float>(4);
    test_transfer<double>(10);
}
