#include <duda/random.hpp>
#include <duda/reductions/reduce_min.hpp>

#include <testing.hpp>

template <typename T>
void test_reduce_min(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const T device_min = duda::reduce_min(x);
    const T host_min = testing::copy(x).minCoeff();

    const T eps = 1e-6;

    REQUIRE(device_min == Approx(host_min).epsilon(eps));
}

TEST_CASE("reduce_min", "[device_matrix][reduce_min]")
{
    test_reduce_min<float>(10, 12);
    test_reduce_min<double>(32, 32);
}
