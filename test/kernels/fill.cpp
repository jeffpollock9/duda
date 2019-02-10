#include <duda/kernels/fill.hpp>

#include <testing.hpp>

template <typename T>
void test_fill(const int rows, const int cols, const T value)
{
    const auto host = testing::host_matrix<T>::Constant(rows, cols, value);

    duda::device_matrix<T> device(rows, cols);
    duda::fill(device, value);

    REQUIRE(testing::all_close(device, host));
}

TEST_CASE("fill matrix", "[device_matrix][fill]")
{
    test_fill<int>(7, 8, 3);
    test_fill<float>(19, 1, 3.14f);
    test_fill<double>(21, 2, -42.0);
}

template <typename T>
void test_fill(const int size, const T value)
{
    const auto host = testing::host_vector<T>::Constant(size, value);

    duda::device_vector<T> device(size);
    duda::fill(device, value);

    REQUIRE(testing::all_close(device, host));
}

TEST_CASE("fill vector", "[device_vector][fill]")
{
    test_fill<int>(17, 4);
    test_fill<float>(11, 3.14f);
    test_fill<double>(2, -42.0);
}
