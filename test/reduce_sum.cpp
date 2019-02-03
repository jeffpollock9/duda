#include <helpers/helpers.hpp>

#include <duda/random.hpp>
#include <duda/reduce_sum.hpp>

template <typename T>
void test_reduce_sum(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const T sum1 = duda::reduce_sum(x.data(), x.size());
    const T sum2 = duda::reduce_sum(x);

    const T eps = 1e-6;

    REQUIRE(sum1 == Approx(copy(x).sum()).epsilon(eps));
    REQUIRE(sum2 == Approx(copy(x).sum()).epsilon(eps));
}

TEST_CASE("reduce_sum", "[device_matrix][reduce_sum]")
{
    test_reduce_sum<float>(10, 12);
    test_reduce_sum<double>(32, 32);
}
