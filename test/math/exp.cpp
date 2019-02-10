#include <duda/math/exp.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_exp(const int rows, const int cols)
{
    const auto x = duda::random_normal<T>(rows, cols);

    const auto device = duda::exp(x);
    const auto host   = testing::copy(x).array().exp().matrix().eval();

    REQUIRE(testing::all_close(device, host));
}

TEST_CASE("exp", "[device_matrix][exp]")
{
    test_exp<float>(4, 4);
    test_exp<float>(4, 7);
}
