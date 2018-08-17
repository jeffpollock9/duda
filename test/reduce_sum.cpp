#include "helpers.hpp"

template <typename T>
void test_reduce_sum(const int rows, const int cols)
{
    device_matrix<T> x_d = duda::random_normal<T>(rows, cols);

    T sum = duda::reduce_sum(x_d.data(), x_d.size());

    host_matrix<T> x_h = copy(x_d);

    const T eps = 1e-4;

    REQUIRE(sum == Approx(x_h.sum()).epsilon(eps));
}

TEST_CASE("fill", "[device_matrix][reduce_sum]")
{
    const int n = 999;

    for (int x = 16; x < n; x += 196)
    {
        test_reduce_sum<float>(x, x * 2);
        test_reduce_sum<double>(x * 4, x);
    }
}
