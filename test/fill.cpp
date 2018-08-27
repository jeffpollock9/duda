#include "helpers.hpp"

#include "fill.hpp"

template <typename T>
void test_fill(const int rows, const int cols, const T value)
{
    host_matrix<T> ans = host_matrix<T>::Constant(rows, cols, value);

    {
        device_matrix<T> x(rows, cols);
        duda::fill(x.data(), x.size(), value);
        REQUIRE(copy(x).isApprox(ans));
    }
    {
        device_matrix<T> x(rows, cols);
        fill(x, value);
        REQUIRE(copy(x).isApprox(ans));
    }
}

TEST_CASE("fill matrix", "[device_matrix][fill]")
{
    const int n = 666;

    for (int x = 3; x < n; x += 42)
    {
        test_fill<int>(x, x + 1, x + 4);
        test_fill<float>(x, x + 1, x + 3.14f);
        test_fill<double>(x, x - 2, x - 42.0);
    }
}

template <typename T>
void test_fill(const int size, const T value)
{
    host_vector<T> ans = host_vector<T>::Constant(size, value);

    {
        device_vector<T> x(size);
        duda::fill(x.data(), x.size(), value);
        REQUIRE(copy(x).isApprox(ans));
    }
    {
        device_vector<T> x(size);
        fill(x, value);
        REQUIRE(copy(x).isApprox(ans));
    }
}

TEST_CASE("fill vector", "[device_vector][fill]")
{
    const int n = 8888;

    for (int x = 89; x < n; x += 42)
    {
        test_fill<int>(x + 17, x + 4);
        test_fill<float>(x + 1, x + 3.14f);
        test_fill<double>(x - 2, x - 42.0);
    }
}
