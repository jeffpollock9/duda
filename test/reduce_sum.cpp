#include "helpers.hpp"

template <typename T>
void test_reduce_sum(const int rows, const int cols)
{
    const device_matrix<T> x = duda::random_normal<T>(rows, cols);

    const T sum1 = duda::reduce_sum(x.data(), x.size());
    const T sum2 = reduce_sum(x);

    const T eps = 1e-4;

    REQUIRE(sum1 == Approx(copy(x).sum()).epsilon(eps));
    REQUIRE(sum2 == Approx(copy(x).sum()).epsilon(eps));
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
