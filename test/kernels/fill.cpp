#include <duda/kernels/fill.hpp>

#include <testing.hpp>

template <typename T>
void test_fill(const int rows, const int cols, const T value)
{
    const auto ans = testing::host_matrix<T>::Constant(rows, cols, value);

    {
        duda::device_matrix<T> x(rows, cols);
        duda::fill(x.data(), x.size(), value);
        REQUIRE(testing::all_close(x, ans));
    }
    {
        duda::device_matrix<T> x(rows, cols);
        duda::fill(x, value);
        REQUIRE(testing::all_close(x, ans));
    }
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
    const auto ans = testing::host_vector<T>::Constant(size, value);

    {
        duda::device_vector<T> x(size);
        duda::fill(x.data(), x.size(), value);
        REQUIRE(testing::all_close(x, ans));
    }
    {
        duda::device_vector<T> x(size);
        duda::fill(x, value);
        REQUIRE(testing::all_close(x, ans));
    }
}

TEST_CASE("fill vector", "[device_vector][fill]")
{
    test_fill<int>(17, 4);
    test_fill<float>(11, 3.14f);
    test_fill<double>(2, -42.0);
}
