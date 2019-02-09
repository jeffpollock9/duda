#include <duda/random.hpp>
#include <duda/reductions/reduce_max.hpp>

#include <testing.hpp>

template <typename T>
void test_reduce_max(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const T device_max = duda::reduce_max(x);
    const T host_max = testing::copy(x).maxCoeff();

    const T eps = 1e-6;

    REQUIRE(device_max == Approx(host_max).epsilon(eps));
}

TEST_CASE("reduce_max", "[device_matrix][reduce_max]")
{
    test_reduce_max<float>(10, 12);
    test_reduce_max<double>(32, 32);
}
