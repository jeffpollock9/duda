#include <duda/math/log.hpp>
#include <duda/random.hpp>

#include <testing.hpp>

template <typename T>
void test_log(const int rows, const int cols)
{
    const auto x = duda::random_uniform<T>(rows, cols);

    const auto device = duda::log(x);
    const auto host   = testing::copy(x).array().log().matrix().eval();

    REQUIRE(testing::all_close(device, host));
}

TEST_CASE("log", "[device_matrix][log]")
{
    test_log<float>(4, 4);
    test_log<double>(16, 4);
}
