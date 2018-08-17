#include "helpers.hpp"

template <typename T>
void test_fill(const int rows, const int cols, const T value)
{
    device_matrix<T> x_d(rows, cols);

    duda::fill(x_d.data(), x_d.size(), value);

    host_matrix<T> x_h = copy(x_d);

    REQUIRE(x_h.isApprox(host_matrix<T>::Constant(rows, cols, value)));
}

TEST_CASE("fill", "[device_matrix][fill]")
{
    const int n = 666;

    for (int x = 3; x < n; x += 42)
    {
        test_fill<float>(x, x + 1, x + 3.14);
        test_fill<double>(x, x - 2, x - 42.0);
    }
}
