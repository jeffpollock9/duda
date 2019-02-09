#include <duda/random.hpp>
#include <duda/reductions/reduce_sum.hpp>

#include <testing.hpp>

template <typename T>
void test_reduce_sum(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const T device_sum1 = duda::reduce_sum(x.data(), x.size());
    const T device_sum2 = duda::reduce_sum(x);

    const T host_sum = testing::copy(x).sum();

    const T eps = 1e-6;

    REQUIRE(device_sum1 == Approx(host_sum).epsilon(eps));
    REQUIRE(device_sum2 == Approx(host_sum).epsilon(eps));
}

TEST_CASE("reduce_sum", "[device_matrix][reduce_sum]")
{
    test_reduce_sum<float>(10, 12);
    test_reduce_sum<double>(32, 32);
}
