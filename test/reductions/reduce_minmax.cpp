#include <duda/random.hpp>
#include <duda/reductions/reduce_minmax.hpp>

#include <testing.hpp>

template <typename T>
void test_reduce_minmax(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const auto device_minmax = duda::reduce_minmax(x);

    const T host_min = testing::copy(x).minCoeff();
    const T host_max = testing::copy(x).maxCoeff();

    const T eps = 1e-6;

    REQUIRE(device_minmax.first == Approx(host_min).epsilon(eps));
    REQUIRE(device_minmax.second == Approx(host_max).epsilon(eps));
}

TEST_CASE("reduce_minmax", "[device_matrix][reduce_minmax]")
{
    test_reduce_minmax<float>(10, 12);
    test_reduce_minmax<double>(32, 32);
}
